import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astroquery.simbad import Simbad
from astroquery.skyview import SkyView
from photutils.isophote import EllipseGeometry, Ellipse

def get_object_coordinates(object_name):
    result_table = Simbad.query_object(object_name)
    ra, dec = result_table["RA"][0], result_table["DEC"][0]
    return SkyCoord(f"{ra} {dec}", unit=(u.hourangle, u.deg))

def get_image_data(coords, survey='DSS', radius=5*u.arcmin):
    image_list = SkyView.get_images(position=coords, survey=survey, radius=radius)
    return image_list[0][0].data, WCS(image_list[0][0].header)

def setup_plot(data, wcs):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': wcs})
    ra, dec = ax.coords
    ra.set_major_formatter("hh:mm:ss")
    dec.set_major_formatter("dd:mm:ss")
    ax.set_xlabel("RA")
    ax.set_ylabel("DEC")
    return fig, ax

def plot_image_and_contours(ax, data):
    sqrt_data = np.sqrt(data + 1)
    ax.imshow(sqrt_data, cmap="gist_heat", origin="lower", aspect="auto")
    
    min_val, max_val = np.min(sqrt_data), np.max(sqrt_data)
    levels = np.linspace(min_val, max_val, 4)[1:-1]  # 2 intermediate levels
    contours = ax.contour(sqrt_data, levels, colors=['yellow', 'cyan'])
    return sqrt_data, contours

def fit_and_plot_ellipse(ax, data, contour_levels):
    center = np.array(data.shape) // 2
    geometry = EllipseGeometry(x0=center[1], y0=center[0], sma=20, eps=0.1, pa=45)
    ellipse = Ellipse(data, geometry)
    isolist = ellipse.fit_image()
    
    for iso in isolist:
        if contour_levels[0] < iso.intens < contour_levels[1]:
            x, y = iso.sampled_coordinates()
            ax.plot(x, y, color='blue')
            
            ax.scatter(center[1], center[0], color='red', s=10, label='Ellipse Center')
            
            angle = np.deg2rad(50)
            radius = iso.sma
            end_x = center[1] + radius * np.cos(angle)
            end_y = center[0] + radius * np.sin(angle)
            ax.plot([center[1], end_x-1.5], [center[0], end_y-1.5], color='white', label='Semi-major Axis')
            
            return radius
    return None

def main():
    # Get M88 coordinates
    m88_coords = get_object_coordinates("M88")
    
    # Get image data
    data, wcs = get_image_data(m88_coords)
    
    # Setup plot
    fig, ax = setup_plot(data, wcs)
    
    # Plot image and contours
    sqrt_data, contours = plot_image_and_contours(ax, data)
    
    # Fit and plot ellipse
    radius = fit_and_plot_ellipse(ax, sqrt_data, contours.levels)
    
    # Finalize plot
    ax.legend()
    plt.show()
    
    if radius:
        print(f"Radius of the ellipse in the blue region: {radius:.2f} pixels")
    else:
        print("Could not determine the ellipse radius.")

if __name__ == "__main__":
    main()
