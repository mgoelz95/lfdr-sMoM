import numpy as np
import csv
from collections import defaultdict
from scipy.sparse import csc_matrix, lil_matrix
import scipy.stats as st

class ProxyDistribution:
    '''Simple proxy distribution to enable specifying signal distributions from the command-line'''
    def __init__(self, name, pdf_method, sample_method):
        self.name = name
        self.pdf_method = pdf_method
        self.sample_method = sample_method

    def pdf(self, x):
        return self.pdf_method(x)

    def sample(self, count=1):
        if count == 1:
            return self.sample_method()
        return np.array([self.sample_method() for _ in range(count)])

    def __repr__(self):
        return self.name

def generate_data_helper(flips, null_mean, null_stdev, signal_dist):
    '''Recursively builds multi-dimensional datasets.'''
    if len(flips.shape) > 1:
        return np.array([generate_data_helper(row, null_mean, null_stdev, signal_dist) for row in flips])

    # If we're on the last dimension, return the vector
    return np.array([signal_dist.sample() if flip else 0 for flip in flips]) + np.random.normal(loc=null_mean, scale=null_stdev, size=len(flips))

def generate_data(null_mean, null_stdev, signal_dist, signal_weights):
    '''Create a synthetic dataset.'''
    # Flip biased coins to decide which distribution to draw each sample from
    flips = np.random.random(size=signal_weights.shape) < signal_weights

    # Recursively generate the dataset
    samples = generate_data_helper(flips, null_mean, null_stdev, signal_dist)

    # Observed z-scores
    z = (samples - null_mean) / null_stdev

    return (z, flips)

def calc_fdr(probs, fdr_level):
    '''Calculates the detected signals at a specific false discovery rate given the posterior probabilities of each point.'''
    pshape = probs.shape
    if len(probs.shape) > 1:
        probs = probs.flatten()
    post_orders = np.argsort(probs)[::-1]
    avg_fdr = 0.0
    end_fdr = 0
    
    for idx in post_orders:
        test_fdr = (avg_fdr * end_fdr + (1.0 - probs[idx])) / (end_fdr + 1.0)
        if test_fdr > fdr_level:
            break
        avg_fdr = test_fdr
        end_fdr += 1

    is_finding = np.zeros(probs.shape, dtype=int)
    is_finding[post_orders[0:end_fdr]] = 1
    if len(pshape) > 1:
        is_finding = is_finding.reshape(pshape)
    return is_finding

def filter_nonrectangular_data(data, filter_value=0):
    '''Convert the square matrix to a vector containing only the values different than the filter values.'''
    x = data != filter_value
    nonrect_vals = np.arange(x.sum())
    nonrect_to_data = np.zeros(data.shape, dtype=int) - 1
    data_to_nonrect = np.where(x.T)
    data_to_nonrect = (data_to_nonrect[1],data_to_nonrect[0])
    nonrect_to_data[data_to_nonrect] = nonrect_vals
    nonrect_data = data[x]
    return (nonrect_data, nonrect_to_data, data_to_nonrect)

def sparse_2d_penalty_matrix(data_shape, nonrect_to_data=None):
    '''Create a sparse 2-d penalty matrix. Optionally takes a map to corrected indices, useful when dealing with non-rectangular data.'''
    row_counter = 0
    data = []
    row = []
    col = []

    if nonrect_to_data is not None:
        for j in range(data_shape[1]):
            for i in range(data_shape[0]-1):            
                idx1 = nonrect_to_data[i,j]
                idx2 = nonrect_to_data[i+1,j]
                if idx1 < 0 or idx2 < 0:
                    continue
                row.append(row_counter)
                col.append(idx1)
                data.append(-1)

                row.append(row_counter)
                col.append(idx2)
                data.append(1.)

                row_counter += 1
        for j in range(data_shape[1]-1):
            for i in range(data_shape[0]):
                idx1 = nonrect_to_data[i,j]
                idx2 = nonrect_to_data[i,j+1]
                if idx1 < 0 or idx2 < 0:
                    continue
                row.append(row_counter)
                col.append(idx1)
                data.append(-1)

                row.append(row_counter)
                col.append(idx2)
                data.append(1.)

                row_counter += 1
    else:
        for j in range(data_shape[1]):
            for i in range(data_shape[0] - 1):
                idx1 = i+j*data_shape[0]
                idx2 = i+j*data_shape[0]+1

                row.append(row_counter)
                col.append(idx1)
                data.append(-1.)

                row.append(row_counter)
                col.append(idx2)
                data.append(1.)

                row_counter += 1

        col_counter = 0
        for i in range(data_shape[0]):
            for j in range(data_shape[1] - 1):
                idx1 = col_counter
                idx2 = col_counter+data_shape[0]

                row.append(row_counter)
                col.append(idx1)
                data.append(-1.)

                row.append(row_counter)
                col.append(idx2)
                data.append(1.)

                row_counter += 1
                col_counter += 1

    num_rows = row_counter
    num_cols = max(col) + 1
    return csc_matrix((data, (row, col)), shape=(num_rows, num_cols))
    
def sparse_1d_penalty_matrix(data_len):
    penalties = np.eye(data_len, dtype=float)[0:-1] * -1
    for i in range(len(penalties)):
        penalties[i,i+1] = 1
    return csc_matrix(penalties)

def cube_trails(xmax, ymax, zmax):
    '''Produces a list of trails following a simple row/col/aisle split strategy for a cube.'''
    trails = []
    for x in range(xmax):
        for y in range(ymax):
            trails.append([x * ymax * zmax + y * zmax + z for z in range(zmax)])
    for y in range(ymax):
        for z in range(zmax):
            trails.append([x * ymax * zmax + y * zmax + z for x in range(xmax)])
    for z in range(zmax):
        for x in range(xmax):
            trails.append([x * ymax * zmax + y * zmax + z for y in range(ymax)])
    return trails

def val_present(data, x, missing_val):
    return missing_val is None or x

def cube_edges(data, missing_val=None):
    '''Produces a list of edges for a cube with potentially missing data.
    If missing_val is specified, entries with that value will be considered
    missing and no edges will be connected to them.'''
    edges = []
    xmax, ymax, zmax = data.shape
    for y in range(ymax):
        for z in range(zmax):
            edges.extend([((x1, y, z), (x2, y, z)) 
                            for x1, x2 in zip(range(data.shape[0]-1), range(1,data.shape[0]))
                            if missing_val is None or (data[x1,y,z] != missing_val and data[x2,y,z] != missing_val)])
    for x in range(xmax):
        for z in range(zmax):
            edges.extend([((x, y1, z), (x, y2, z))
                            for y1, y2 in zip(range(data.shape[1]-1), range(1,data.shape[1]))
                            if missing_val is None or (data[x,y1,z] != missing_val and data[x,y2,z] != missing_val)])
    for x in range(xmax):
        for y in range(ymax):
            edges.extend([((x, y, z1), (x, y, z2)) 
                            for z1, z2 in zip(range(data.shape[2]-1), range(1,data.shape[2]))
                            if missing_val is None or (data[x,y,z1] != missing_val and data[x,y,z2] != missing_val)])
    return edges

def cube_trails_missing_helper(data, trails, cur_trail, v1, v2, missing_val):
    if data[v1] == missing_val or data[v2] == missing_val:
        if len(cur_trail) > 1:
            trails.append(cur_trail)
            cur_trail = []
    else:
        if len(cur_trail) == 0:
            cur_trail.append(v1)
        cur_trail.append(v2)
    return cur_trail

def cube_trails_missing(data, missing_val=None):
    '''Generates row/col/aisle trails for a cube when there may be missing data.'''
    trails = []
    xmax, ymax, zmax = data.shape
    for y in range(ymax):
        for z in range(zmax):
            cur_trail = []
            for x1, x2 in zip(range(data.shape[0]-1), range(1,data.shape[0])):
                v1 = (x1,y,z)
                v2 = (x2,y,z)
                cur_trail = cube_trails_missing_helper(data, trails, cur_trail, v1, v2, missing_val)
            if len(cur_trail) > 1:
                trails.append(cur_trail)
                
    for x in range(xmax):
        for z in range(zmax):
            cur_trail = []
            for y1, y2 in zip(range(data.shape[1]-1), range(1,data.shape[1])):
                v1 = (x,y1,z)
                v2 = (x,y2,z)
                cur_trail = cube_trails_missing_helper(data, trails, cur_trail, v1, v2, missing_val)
            if len(cur_trail) > 1:
                trails.append(cur_trail)

    for x in range(xmax):
        for y in range(ymax):
            cur_trail = []
            for z1, z2 in zip(range(data.shape[2]-1), range(1,data.shape[2])):
                v1 = (x, y, z1)
                v2 = (x, y, z2)
                cur_trail = cube_trails_missing_helper(data, trails, cur_trail, v1, v2, missing_val)
            if len(cur_trail) > 1:
                trails.append(cur_trail)
                            
    return trails


def load_trails(filename):
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        return load_trails_from_reader(reader)

def load_trails_from_reader(reader):
    trails = []
    breakpoints = []
    edges = defaultdict(list)
    for line in reader:
        if len(trails) > 0:
            breakpoints.append(len(trails))
        nodes = [int(x) for x in line]
        trails.extend(nodes)
        for n1,n2 in zip(nodes[:-1], nodes[1:]):
            edges[n1].append(n2)
            edges[n2].append(n1)
    if len(trails) > 0:
        breakpoints.append(len(trails))
    return (len(breakpoints), np.array(trails, dtype="int32"), np.array(breakpoints, dtype="int32"), edges)

def save_trails(filename, trails):
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(trails)

def pretty_str(p, decimal_places=2):
    '''Pretty-print a matrix or vector.'''
    if len(p.shape) == 1:
        return vector_str(p, decimal_places)
    if len(p.shape) == 2:
        return matrix_str(p, decimal_places)
    raise Exception('Invalid array with shape {0}'.format(p.shape))

def matrix_str(p, decimal_places=2):
    '''Pretty-print the matrix.'''
    return '[{0}]'.format("\n  ".join([vector_str(a, decimal_places) for a in p]))

def vector_str(p, decimal_places=2):
    '''Pretty-print the vector values.'''
    style = '{0:.' + str(decimal_places) + 'f}'
    return '[{0}]'.format(", ".join([style.format(a) for a in p]))

def mean_filter(pvals, edges, rescale=True):
    '''Given a list of p-values and their neighbors, applies a mean filter
    that replaces each p_i with p*_i where p*_i = mean(neighbors(p_i)).
    If rescale is true, then the p-values are rescaled to be variance 1.'''
    return np.array([np.mean(pvals[edges[i] + [i]]) * (np.sqrt(len(edges[i]) + 1) if rescale else 1) for i,p in enumerate(pvals)])

def median_filter(pvals, edges):
    '''Given a list of p-values and their neighbors, applies a median filter
    that replaces each p_i with p*_i where p*_i = median(neighbors(p_i)).'''
    return np.array([np.median(pvals[edges[i] + [i]]) for i,p in enumerate(pvals)])

def _local_agg_fdr_helper(fdr_level, p_star, ghat, ghat_lambda, wstar_lambda, tmin, tmax, tmin_fdr, tmax_fdr, rel_tol=1e-4):
    '''Finds the t-level via binary search.'''
    if np.isclose(tmin, tmax, atol=rel_tol) or np.isclose(tmin_fdr, tmax_fdr, atol=rel_tol) or tmax_fdr <= fdr_level:
        return (tmax, tmax_fdr) if tmax_fdr <= fdr_level else (tmin, tmin_fdr)
    tmid = (tmax + tmin) / 2.
    tmid_fdr = wstar_lambda * ghat(p_star, tmid) / (max(1,(p_star < tmid).sum()) * (1-ghat_lambda))
    print('t: [{0}, {1}, {2}] => fdr: [{3}, {4}, {5}]'.format(tmin, tmid, tmax, tmin_fdr, tmid_fdr, tmax_fdr))
    if tmid_fdr <= fdr_level:
        return _local_agg_fdr_helper(fdr_level, p_star, ghat, ghat_lambda, wstar_lambda, tmid, tmax, tmid_fdr, tmax_fdr)
    return _local_agg_fdr_helper(fdr_level, p_star, ghat, ghat_lambda, wstar_lambda, tmin, tmid, tmin_fdr, tmid_fdr)

def local_agg_fdr(pvals, edges, fdr_level, lmbda = 0.1):
    '''Given a list of p-values and the graph connecting them, applies a median
    filter to locally aggregate them and then performs a corrected FDR procedure
    from Zhang, Fan, and Yu (Annals of Statistics, 2011). lmbda is a tuning
    constant typically set to 0.1.'''
    p_star = median_filter(pvals, edges) # aggregate p-values
    ghat = lambda p, t: (p >= (1-t)).sum() / max(1., (2.0 * (p > 0.5).sum() + (p==0.5).sum())) # empirical null CDF
    wstar_lambda = (p_star > lmbda).sum() # number of nonrejects at the level lambda
    ghat_lambda = ghat(p_star, lmbda) # empirical null CDF at rejection level lambda    
    # Use binary search to find the highest t value that satisfies the fdr level
    tmin = 0.
    tmax = 1.
    tmin_fdr = wstar_lambda * ghat(p_star, tmin) / (max(1,(p_star < tmin).sum()) * (1-ghat_lambda))
    tmax_fdr = wstar_lambda * ghat(p_star, tmax) / (max(1,(p_star < tmax).sum()) * (1-ghat_lambda))
    t, tfdr = _local_agg_fdr_helper(fdr_level, p_star, ghat, ghat_lambda, wstar_lambda, tmin, tmax, tmin_fdr, tmax_fdr)
    print('t: {0} tfdr: {1}'.format(t, tfdr))
    # Returns the indices of all discoveries
    return np.where(p_star < t)[0]

def p_value(z, mu0=0., sigma0=1.):
    return 2*(1.0 - st.norm.cdf(np.abs((z - mu0) / sigma0)))

def benjamini_hochberg(z, fdr, mu0=0., sigma0=1.):
    '''Performs Benjamini-Hochberg multiple hypothesis testing on z at the given false discovery rate threshold.'''
    z_shape = z.shape if len(z.shape) > 1 else None
    if z_shape is not None:
        z = z.flatten()
    p = p_value(z, mu0=mu0, sigma0=sigma0)
    p_orders = np.argsort(p)
    discoveries = []
    m = float(len(p_orders))
    for k, s in enumerate(p_orders):
        if p[s] <= (k+1) / m * fdr:
            discoveries.append(s)
        else:
            break
    discoveries = np.array(discoveries)
    if z_shape is not None:
        x = np.zeros(z.shape)
        x[discoveries] = 1
        discoveries = np.where(x.reshape(z_shape) == 1)
    return discoveries
