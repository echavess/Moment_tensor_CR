import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np 
from obspy.imaging.beachball import beachball, beach
from numpy.linalg import eig

import warnings
warnings.simplefilter('ignore')

def meca_angles(catalog):
    # Load the catalog
    catalog_data = pd.read_csv(catalog, sep=' ')
    catalog_data = catalog_data.apply(pd.to_numeric, errors='coerce')

    fig, axes= plt.subplots(2, 3, figsize=(11,10))
    ax1 = axes[0,0]
    ax2 = axes[0,1]
    ax3 = axes[0,2]
    ax4 = axes[1,0]
    ax5 = axes[1,1]
    ax6 = axes[1,2]

    sb.histplot(data=catalog_data, x="Strike",  kde=True, ax=ax1, color='b')
    sb.histplot(data=catalog_data, x="Dip",  kde=True, ax=ax2, color='g')
    sb.histplot(data=catalog_data, x="Rake",  kde=True, ax=ax3, color='r')

    # Convert angles to radians for plotting
    catalog_data['Strike1_rad'] = np.deg2rad(catalog_data['Strike'])
    catalog_data['Dip1_rad'] = np.deg2rad(catalog_data['Dip'])
    catalog_data['Rake1_rad'] = np.deg2rad(catalog_data['Rake'])

    fig.delaxes(ax4)
    ax4 = plt.subplot(2, 3, 4, polar=True)
    ax4.hist(catalog_data['Strike1_rad'], bins=36, color='b', edgecolor='black')
    #ax4.set_title("Strike Angles", fontsize=15)

    # Dip angle rose diagram
    fig.delaxes(ax5) 
    ax5 = plt.subplot(2, 3, 5, polar=True)
    ax5.hist(catalog_data['Dip1_rad'], bins=36, color='g', edgecolor='black')
    #ax5.set_title("Dip Angles", fontsize=15)

    # Rake angle rose diagram
    fig.delaxes(ax6)
    ax6 = plt.subplot(2, 3, 6, polar=True)
    ax6.hist(catalog_data['Rake1_rad'], bins=36, color='r', edgecolor='black')
    #ax6.set_title("Rake Angles", fontsize=15)
    fig.savefig("Moment_tensor_CR_angles.png", dpi=600, format='png')

def stress_inversion(catalog):

    catalog_data = pd.read_csv(catalog, sep=' ', names=["Latitude", "Longitude", "Depth", "Magnitude", "Strike", "Dip", "Rake"])
    catalog_data = catalog_data.apply(pd.to_numeric, errors='coerce')
    catalog_data_clean = catalog_data.dropna(subset=['Strike', 'Dip', 'Rake'])

    def strike_dip_rake_to_moment_tensor(strike, dip, rake):
        # Convert angles to radians
        strike_rad = np.deg2rad(strike)
        dip_rad = np.deg2rad(dip)
        rake_rad = np.deg2rad(rake)
        # Compute the moment tensor components
        Mrr = np.sin(2 * dip_rad) * np.sin(rake_rad)
        Mtt = -np.sin(dip_rad) * np.cos(rake_rad) * np.sin(2 * strike_rad) - np.sin(2 * dip_rad) * np.sin(rake_rad)
        Mpp = np.sin(dip_rad) * np.cos(rake_rad) * np.cos(2 * strike_rad)
        Mrt = -np.sin(dip_rad) * np.cos(rake_rad) * np.sin(strike_rad) + np.cos(dip_rad) * np.sin(rake_rad) * np.cos(strike_rad)
        Mrp = -np.sin(dip_rad) * np.cos(rake_rad) * np.cos(strike_rad) - np.cos(dip_rad) * np.sin(rake_rad) * np.sin(strike_rad)
        Mtp = -np.cos(dip_rad) * np.sin(rake_rad)

        # Moment tensor matrix
        moment_tensor = np.array([[Mrr, Mrt, Mrp],
                                [Mrt, Mtt, Mtp],
                                [Mrp, Mtp, Mpp]])
        
        return moment_tensor

    def stress_inversion(moment_tensors):
        stress_axes = []
        for mt in moment_tensors:
            evals, evecs = eig(mt)
            # Sort eigenvalues and associated eigenvectors
            idx = np.argsort(evals)[::-1]  # descending order
            evals = evals[idx]
            evecs = evecs[:, idx]
            stress_axes.append((evals, evecs))  # (eigenvalues, eigenvectors)
        return stress_axes


    # Convert strike, dip, and rake to moment tensors for the cleaned catalog
    moment_tensors_clean = catalog_data_clean.apply(lambda row: strike_dip_rake_to_moment_tensor(row['Strike'], row['Dip'], row['Rake']), axis=1)

    # Perform stress inversion on the cleaned data
    stress_axes_clean = stress_inversion(moment_tensors_clean)

    # Extract sigma1, sigma2, sigma3 orientations from the results
    sigma1_orientations_clean = np.array([axis[1][:, 0] for axis in stress_axes_clean])  # Eigenvector corresponding to largest eigenvalue
    sigma2_orientations_clean = np.array([axis[1][:, 1] for axis in stress_axes_clean])
    sigma3_orientations_clean = np.array([axis[1][:, 2] for axis in stress_axes_clean])

    #  Plot sigma1, sigma2, and sigma3 directions in a stereonet-like plot
    fig, ax = plt.subplots(1, 3, figsize=(7,5), subplot_kw={'projection': 'polar'})

    ax1 = ax[0]
    ax2 = ax[1]
    ax3 = ax[2]

    ax1.set_title("Sigma1")
    ax1.scatter(np.arctan2(sigma1_orientations_clean[:, 1], sigma1_orientations_clean[:, 0]), 
                np.abs(sigma1_orientations_clean[:, 2]), color='r', lw=0.5, edgecolor='k', alpha=0.6, s=10)

    # # Plot sigma2
    ax2.set_title("Sigma2")
    ax2.scatter(np.arctan2(sigma2_orientations_clean[:, 1], sigma2_orientations_clean[:, 0]), 
                np.abs(sigma2_orientations_clean[:, 2]), color='g', lw=0.5, edgecolor='k', alpha=0.6)

    # # Plot sigma3
    ax3.set_title("Sigma3")
    ax3.scatter(np.arctan2(sigma3_orientations_clean[:, 1], sigma3_orientations_clean[:, 0]), 
                np.abs(sigma3_orientations_clean[:, 2]), color='b', lw=0.5, edgecolor='k', alpha=0.6)
    
    fig.savefig("Stress_orientations.png", format='png', dpi=600)
    

def mapping(catalog):

    catalog_data = pd.read_csv(catalog, sep=' ', names=["Latitude", "Longitude", "Depth", "Magnitude", "Strike", "Dip", "Rake"])
    catalog_data = catalog_data.apply(pd.to_numeric, errors='coerce')

    fig, ax = plt.subplots(1,1,figsize=(8,8))
    for i, stk in enumerate(catalog_data['Strike']):
        dp = catalog_data['Dip'][i]
        rk = catalog_data['Rake'][i]
        x = catalog_data['Longitude'][i]
        y = catalog_data['Latitude'][i]
        np1 = [stk, dp, rk]

        if any(np.isnan(x) for x in np1):
            print("List contains NaN, skipping processing.")
        else:
            print(f"Processing the list: {np1}")
            meca = beach(np1, xy=(x,y), width=catalog_data['Magnitude'][i]/35, linewidth=0.5, zorder=100)
            print(meca)
            ax.scatter(x,y,s=0.1, c='k')
            ax.add_collection(meca)
            ax.set_aspect("equal")


catalog='combined_catalog_MOMENT_TENSORS.txt'
meca_angles(catalog=catalog)
stress_inversion(catalog=catalog)
mapping(catalog=catalog)

plt.tight_layout()
plt.show()