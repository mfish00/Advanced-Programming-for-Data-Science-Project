get_ipython().system('pip install download')

from datetime import datetime
import sys
import os
import urllib.request
import itertools
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import geopandas as gpd

sys.path.append("./Methods")
plt.style.use('seaborn')


class Agros_class:
    def __init__(self):
        '''
        Init method of the class Agros_class
        Parameters
        nothing
        ------------
        Returns
        nothing
        ---------

        '''
        self.data = None
        self.file = False
        self.download_file()
        self.read_data()

        # Dictionary for merging countries
        self.merge_dict = {
            'United States': 'United States of America',
            'Dominican Republic': 'Dominica',
            'Democratic Republic of Congo': 'Democratic Republic of the Congo',
            'Congo': 'Republic of the Congo',
            'Bahamas': 'The Bahamas',
            'Tanzania': 'United Republic of Tanzania',
            'Eswatini': 'eSwatini'
        }

    def download_file(self):
        '''
        Download data file into downloads/ directory
        Parameters
        nothing
        ------------
        Returns
        nothing
        ---------
        '''
        url = "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Agricultural%20total%20factor%20productivity%20(USDA)/Agricultural%20total%20factor%20productivity%20(USDA).csv"
        file_path = "downloads/data_file.csv"
        
        geo_url = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/cultural/ne_50m_admin_0_countries.zip"
        geo_path = "downloads/ne_50m_admin_0_countries.zip"
        
        if not os.path.exists("downloads"):
            os.makedirs("downloads")
        
        if os.path.exists(file_path):
            print("Agri File already exists, skipping download")
        else:
            print("Downloading Agri file from URL...")
            urllib.request.urlretrieve(url, file_path)
            print("Agri File downloaded successfully")
            
            
        try:
            if not os.path.exists(geo_path):
                req = urllib.request.Request(
                    geo_url,
                    data=None,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
                    }
                )
                with urllib.request.urlopen(req) as response, open(geo_path, 'wb') as out_file:
                    out_file.write(response.read())
                print("Geo File downloaded successfully")
            else:
                print("Geo File already exists, skipping download")
        except Exception as e:
            print(f"Error downloading Geo file: {e}")
        
        self.file = True

    def read_data(self):
        '''
        Read data file into a Pandas DataFrame
        Parameters
        nothing
        ------------
        Returns
        nothing
        ---------
        '''
        file_path = "downloads/data_file.csv"
        if os.path.exists(file_path):
            self.data = pd.read_csv(file_path)
            print("Agri Data loaded successfully into a Pandas DataFrame")
        else:
            print("Agri Data file not found")
            
        geo_path = "downloads/ne_50m_admin_0_countries.zip"    
        if os.path.exists(geo_path):
            shpfile = "zip://downloads/ne_50m_admin_0_countries.zip"
            self.geo = gpd.read_file(shpfile)
            print("Geo Data loaded successfully into a GeoPandas DataFrame")
        else:
            print("Geo Data file not found")

    def get_countries(self):
        '''
        Get a list of the available countries in the data set
        Parameters
        nothing
        ------------
        Returns
        A list of the available countries in the data set
        ---------
        '''
        if self.data is not None:
            return list(self.data['Entity'].unique())
        else:
            print("Data not loaded yet")
    
    def plot_quantity_correlation(self):
        '''
        Plot a correlation matrix between the "_quantity" columns in the dataset
        Parameters
        ----------
        nothing
        ------------
        Returns
        ----------
        nothing
        '''
        if self.data is not None:
            # Select only columns that contain "_quantity"
            quantity_cols = [col for col in self.data.columns if '_quantity' in col]
            # Compute correlation matrix
            corr = self.data[quantity_cols].corr()
            # Plot heatmap of correlation matrix
            sns.heatmap(corr, annot=True, cmap='coolwarm')
            plt.show()
        else:
            print("Data not loaded yet")
    
    def plot_output_area_chart(self, country=None, normalize=False):
        '''
        Plot an area chart of the distinct "_output_" columns.
        '''
        if self.data is None:
            print("Data not loaded yet")
            return

        if country is None or country == 'World':
            data_to_plot = self.data.groupby('Year').sum().loc[:, 'ag_land_quantity':'irrigation_quantity']
        else:
            if country not in self.get_countries():
                raise ValueError(f"{country} does not exist in the dataset")

            data_to_plot = self.data[self.data['Entity'] == country].set_index('Year').loc[:, 'ag_land_quantity':'irrigation_quantity']

        if normalize:
            data_to_plot = data_to_plot.divide(data_to_plot.sum(axis=1), axis=0) * 100
            
        data_to_plot.plot.area()
        plt.xlabel("Year")
        plt.ylabel("Output")
        plt.title(f"{'World' if country is None else country} Output{' (Normalized)' if normalize else ''}")
        plt.show()
    
    def compare_output_total(self, countries):
        '''
        Compare the total output of selected countries across years and plot it
        Parameters
        ----------
        countries : str or list
            The name of a country or a list of country names to compare
        ------------
        Returns
        ----------
        nothing
        '''
        if self.data is None:
            print("Data not loaded yet")
            return
        
        # Convert string to list if only one country is provided
        if isinstance(countries, str):
            countries = [countries]
        
        # Filter data for selected countries
        filtered_data = self.data[self.data['Entity'].isin(countries)]
        
        # Compute total output for each year
        data_to_plot = filtered_data.groupby('Year').sum().loc[:, 'output':'output_quantity']
        data_to_plot['Entity'] = data_to_plot.sum(axis=1)
        
        plt.plot(data_to_plot.index, data_to_plot['Entity'], label= countries)
        plt.legend()
        plt.xlabel('Year')
        plt.ylabel('Output (US$)')
        plt.show()
    
    def gapminder(self, year):
        '''
        Plot a scatter plot comparing fertilizer_quantity and output_quantity for a given year
        The area of each dot represents a third relevant variable
        Parameters
        ----------
        year: int
        The year to plot the scatter plot for
        ------------
        Returns
        ----------
        nothing
        '''
        # Check if year is an integer
        if not isinstance(year, int):
            raise TypeError("Year must be an integer")
            
        # Filter data for given year
        data_filtered = self.data[self.data['Year'] == year]
            
        # Set area size based on a third relevant variable (here, 'land_area' column)
        area_size = data_filtered['ag_land_quantity']/1000
        
        # Plot scatter plot
        plt.scatter(data_filtered['fertilizer_quantity'], data_filtered['output_quantity'], s=area_size,  alpha=0.4, edgecolors='grey', linewidth=2)

        # Set plot title and axis labels
        plt.title(f"Gapminder Plot for year {year}")
        plt.xlabel("Fertilizer Quantity")
        plt.ylabel("Output Quantity")
        plt.xscale("log")
        plt.yscale("log")
            
        # Show plot
        plt.show()

    def choropleth(self, year):
        """
        Creates a choropleth map of the world for a given year.
    
        Parameters
        ----------
        year : int
            The year for which the map should be created.

        Returns
        -------
        A choropleth map of the world for a given year.
        """
    
        if not isinstance(year, int):
            raise ValueError("Year must be an integer.")
        
        # Filter data for given year
        self.data = self.data[self.data['Year'] == year]
        
        # Rename country names to match the ones in the geographical data
        self.data['Entity'] = self.data['Entity'].replace(self.merge_dict) 
        
        # Read geographical data
        geo_data = self.geo[['ADMIN', 'geometry']]
                
        # Merge dataframes
        merge_data = geo_data.merge(self.data, left_on='ADMIN', right_on='Entity')
        
        # Plot choropleth map
        fig, ax = plt.subplots(1, figsize=(20, 10))
        ax.set_title(f"Agricultural total factor productivity in {year}", fontsize=20)
        merge_data.plot(column='tfp', cmap='Accent', ax=ax, legend=True,
                        legend_kwds={'label': "Agricultural total factor productivity"})
        plt.show()
        