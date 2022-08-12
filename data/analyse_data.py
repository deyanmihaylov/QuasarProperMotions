import sys

import pandas as pd
import numpy as np
import scipy.linalg

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def covariant_matrix(errors, corr):
    """
    Function for computing the covariant matrix from errors and correlations  
    """
    covariant_matrix = np.einsum('...i,...j->...ij', errors, errors)
    covariant_matrix[...,0,1] = covariant_matrix[...,1,0] = np.multiply(covariant_matrix[...,1,0], corr.flatten())
    return covariant_matrix

def histogram_plot(path, file_name, bins, counts, mode, figsize, plot_title, x_label, y_label):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    width = 0.8 * bins[1]

    bin_marks = np.array(bins[:-1]) + 0.5*bins[1]

    if mode == 'v':
        ax.set_xlim(0.0, bins[-1])
        ax.set_xticks(bins)
        ax.set_xticklabels([float('%.3g' % b) for b in bins], rotation=45)

        y_lim_min = log_floor(np.min(counts[np.nonzero(counts)]))
        y_lim_max = log_ceil(np.max(counts))

        ax.set_ylim(y_lim_min, y_lim_max)

        ax.semilogy()

        ax.bar(bin_marks, counts, width=width, align='center')
    elif mode == 'h':
        x_lim_min = log_floor(np.min(counts[np.nonzero(counts)]))
        x_lim_max = log_ceil(np.max(counts))
        ax.set_xlim(x_lim_min, x_lim_max)
    
        ax.set_ylim(0.0, bins[-1])
        ax.set_yticks(bins)
        ax.set_yticklabels([float('%.3g' % b) for b in bins])

        ax.invert_yaxis()

        ax.semilogx()

        ax.barh(y=bin_marks, width=counts, height=width, align='center')

    ax.set_xlabel(x_label, fontdict=None, labelpad=None, fontsize=16)
    ax.set_ylabel(y_label, fontdict=None, labelpad=None, fontsize=16)

    ax.set_title(plot_title, fontdict=None, loc='center', pad=None, fontsize=16)

    plt.savefig("hist.png")

def eccentricity_of_ellipse(minor_axis, major_axis):
    return np.sqrt(1 - (minor_axis / major_axis)**2)

def minor_major_axes(axis1, axis2):
    if axis1 < axis2:
        return axis1, axis2
    else:
        return axis2, axis1

def log_ceil(x):
    log_ceil = 10.0 ** np.ceil(np.log10(x))

    return 10. * log_ceil if is_power (10, x) else log_ceil

def log_floor(x):
    log_floor =  10.0 ** np.floor(np.log10(x))

    return 0.1 * log_floor if is_power (10, x) else log_floor

def is_power(x, y):
    res_int = int(np.log(y) / np.log(x))
 
    res_float = np.log(y) / np.log(x)

    return 1 if res_int == res_float else 0


filename = str(sys.argv[1])

data = pd.read_csv(filename)

print(f"Dataset: {filename}")

print(f"Size of the raw dataset: {len(data)}")

dropna_columns = [
    'ra',
    'dec',
    'pmra',
    'pmdec',
    'pmra_error',
    'pmdec_error',
    'pmra_pmdec_corr',
]

# remove rows in the dataset for which one of these columns is NaN
data.dropna(
    axis=0,
    how='any',
    thresh=None,
    subset=dropna_columns,
    inplace=True
)

print(f"Size of the dataset after dropping NaNs: {len(data)}")

# Get the QSO positions in ra and dec
positions = np.deg2rad(data[['ra', 'dec']].values)

# Get the QSO proper motions 
proper_motions = data[['pmra', 'pmdec']].values

# Get the QSO proper motion errors
proper_motions_errors = data[['pmra_error', 'pmdec_error']].values
proper_motions_errors[:,0] = (
    proper_motions_errors[:,0] / np.cos(positions[:,1])
)

# Get the QSO proper motion error correlations
proper_motions_error_corrs = data[['pmra_pmdec_corr']].values

# Compute the covariance matrices for each QSO
covariance = covariant_matrix(
    proper_motions_errors,
    proper_motions_error_corrs
)

# Take the square root of each covariance matrix
matrix_sqrt = np.vectorize(scipy.linalg.sqrtm, signature="(m,n)->(m,n)")
sqrt_covariance = matrix_sqrt(covariance)

# Compute the eigenvalues and eigenvectors for each QSO
eigenvalues, eigenvectors = np.linalg.eig(sqrt_covariance)
eigenvectors = np.transpose(eigenvectors, axes=[0,2,1])

eigenvalue_norms = np.linalg.norm(eigenvalues, axis=1)
norms_counts, norms_bins = np.histogram(
    eigenvalue_norms,
    bins = 30,
    range=(0.0, eigenvalue_norms.max())
)

# Calculate eccentricities of covariance ellipses
eigenvalues_sorted = np.sort(eigenvalues)[:,::-1]
eccentricities = eccentricity_of_ellipse(
    eigenvalues_sorted[:, 1],
    eigenvalues_sorted[:, 0]
)

print(f"The min covariance ellipse eccentricity is {np.min(eccentricities)}.")
print(f"The max covariance ellipse eccentricity is {np.max(eccentricities)}.")

count = 0
rads = np.array([])

for i, eigs in enumerate(eigenvalues):
    if eigs[0] > eigs[1]:
        eig1 = eigs[0]
        eig2 = eigs[1]
        ev1 = eigenvectors[i, 0]
        ev2 = eigenvectors[i, 1]
    else:
        eig1 = eigs[1]
        eig2 = eigs[0]
        ev1 = eigenvectors[i, 1]
        ev2 = eigenvectors[i, 0]
                
    angle = np.rad2deg(np.arctan2(ev1[1], ev1[0]))
    
    cos_angle = np.cos(np.radians(180 - angle))
    sin_angle = np.sin(np.radians(180 - angle))

    xct = (
        proper_motions[i, 0] * cos_angle
        - proper_motions[i, 1] * sin_angle
    )
    yct = (
        proper_motions[i, 0] * sin_angle
        + proper_motions[i, 1] * cos_angle
    )
    
    rad_cc = np.sqrt((xct/eig1)**2 + (yct/eig2)**2)
    rads = np.append(rads, rad_cc)

max_rad_cc_id = np.argmax(rads)

if eigenvalues[max_rad_cc_id, 0] > eigenvalues[max_rad_cc_id, 1]:
    eig1 = eigenvalues[max_rad_cc_id, 0]
    eig2 = eigenvalues[max_rad_cc_id, 1]

    ev1 = eigenvectors[max_rad_cc_id, 0]
    ev2 = eigenvectors[max_rad_cc_id, 1]
else:
    eig1 = eigenvalues[max_rad_cc_id, 1]
    eig2 = eigenvalues[max_rad_cc_id, 0]

    ev1 = eigenvectors[max_rad_cc_id, 1]
    ev2 = eigenvectors[max_rad_cc_id, 0]

angle = np.rad2deg(np.arctan2(ev1[1], ev1[0]))

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_xlim(
    -1.1 * proper_motions[max_rad_cc_id, 0],
    1.1 * proper_motions[max_rad_cc_id, 0]
)
ax.set_ylim(
    -1.1 * proper_motions[max_rad_cc_id, 0],
    1.1 * proper_motions[max_rad_cc_id, 0]
)

ax.arrow(
    0., 0.,
    proper_motions[max_rad_cc_id, 0], proper_motions[max_rad_cc_id, 1],
    width=0.01, color='black'
)
ax.arrow(0., 0., ev1[0], ev1[1], width=0.01, color='blue')
ax.arrow(0., 0., ev2[0], ev2[1], width=0.01, color='red')

ell1 = patches.Ellipse(
    (0., 0.),
    2*eig1, 2*eig2,
    angle=angle,
    fill=False,
    edgecolor='green',
    linewidth=2
)

ax.add_patch(ell1)

plt.savefig("max_rad_cc.png")

print(f"The min ratio of PM / covariance radius is {np.min(rads)}.")
print(f"The max ratio of PM / covariance radius is {np.max(rads)}.")

rad_counts, rad_bins = np.histogram(
    rads,
    bins=2 * int(np.ceil(np.max(rads))),
    range=(0.0, 20)
)
fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(111)

width = rad_bins[1]

bin_marks = np.array(rad_bins[:-1]) + 0.5 * rad_bins[1]

ax.set_xlim(0.0, 6)
ax.set_xticks(rad_bins)
ax.set_xticklabels([float('%.3g' % b) for b in rad_bins], rotation=45)

y_lim_min = log_floor(np.min(rad_counts[np.nonzero(rad_counts)]))
y_lim_max = log_ceil(np.max(rad_counts))

ax.set_ylim(y_lim_min, y_lim_max)

ax.semilogy()

ax.bar(
    bin_marks,
    rad_counts,
    width=width,
    align="center",
    edgecolor="blue",
)

x_label="Number of sigma"
y_label="Number of QSOs"
ax.set_xlabel(x_label, fontdict=None, labelpad=None, fontsize=16)
ax.set_ylabel(y_label, fontdict=None, labelpad=None, fontsize=16)

plot_title="Proper motions vs errors histogram"
ax.set_title(plot_title, fontdict=None, loc='center', pad=None, fontsize=16)

logL_xs = np.linspace(0., 20., 201)
# quadratic_logL_ys = quadratic_logL(logL_xs)
# permissive_logL_ys = permissive_logL(logL_xs)
# permissive_logL_ys[0] = 0

# quadratic_logL_ys = quadratic_logL_ys / quadratic_logL_ys.max()
# permissive_logL_ys = permissive_logL_ys / permissive_logL_ys.max()

# ax.plot(logL_xs, quadratic_logL_ys*rad_counts.max(), c="r")
# ax.plot(logL_xs, permissive_logL_ys*rad_counts.max(), c="g")

# ax2.plot(logL_xs, quadratic_logL_ys, c="r")
# ax2.plot(logL_xs, permissive_logL_ys, c="g")

# q_mean = 0.797885
# p_mean = 2.86875

plt.savefig("radii_hist.png")

# Bin the eccentricities
ecc_counts, ecc_bins = np.histogram(
    eccentricities,
    bins = 20,
    range=(0.0, 1.0)
)
ecc_median = np.median(eccentricities)

fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(111)

width = ecc_bins[1]

bin_marks = np.array(ecc_bins[:-1]) + 0.5*ecc_bins[1]

ax.set_xlim(0.0, ecc_bins[-1])
ax.set_xticks(ecc_bins)
ax.set_xticklabels([float('%.3g' % b) for b in ecc_bins], rotation=45)

y_lim_min = log_floor(np.min(ecc_counts[np.nonzero(ecc_counts)]))
y_lim_max = log_ceil(np.max(ecc_counts))

ax.set_ylim(y_lim_min, y_lim_max)

ax.semilogy()

ax.bar(
    bin_marks,
    ecc_counts,
    width=width,
    align="center",
    edgecolor="blue",
)

ax.set_xlabel("Eccentricity", fontdict=None, labelpad=None, fontsize=16)
ax.set_ylabel("Number of QSOs", fontdict=None, labelpad=None, fontsize=16)

ax.set_title("Eccentricity histogram", fontdict=None, loc='center', pad=None, fontsize=16)

plt.savefig("eccentricities.png")

plt.clf()
fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(111, projection='mollweide')

ra = np.array([x - 2 * np.pi if x > np.pi else x for x in positions[:, 0]])
dec = positions[:, 1]

colors = plt.cm.Blues(eccentricities)

# plot the positions
im = ax.scatter(ra, dec, c=eccentricities, cmap=plt.cm.brg, s=0.05)

fig.colorbar(im)
plt.tight_layout()
plt.savefig("eccentricities_sky.png")


rads_log = rads.copy()
rads_log -= rads_log.min()
rads_log = rads_log / rads_log.max()

plt.clf()
fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(111, projection='mollweide')

ra = np.array([x - 2 * np.pi if x > np.pi else x for x in positions[:, 0]])
dec = positions[:, 1]

# plot the positions
im = ax.scatter(ra, dec, c=rads_log, cmap=plt.cm.brg, s=0.1)

plt.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False
)
plt.tight_layout()
plt.savefig("positions.png")
