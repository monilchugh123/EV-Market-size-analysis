#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('Electric_Vehicle_Population_Data.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


a = [1, 2, 3]
b = [4, 5, 6]

print(a + b)


# # analyzing the EV Adoption Over Time

# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


ev_adoption_by_year = df['Model Year'].value_counts().sort_index()
sns.set_style('whitegrid')
plt.figure(figsize=(12,6))
sns.barplot(x = ev_adoption_by_year.index, y = ev_adoption_by_year.values, palette='viridis')
plt.title('EV sdoption over time')
plt.xlabel('Year')
plt.ylabel('Number of vehicles registered')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print('Conclusion: From the above bar chart , it is clear that EV sales has been increasing over time , especially noting a significant upward trend starting around 2016')


# # Geographical distribution

# In[11]:


ev_country_dist = df['Country'].value_counts()
top_countries = ev_country_dist.head(3).index
# filtering the dataset for these top counties
top_countries_data = df[df['Country'].isin(top_countries)]
# analysing the distribution of EVs within the cities
ev_city_dist_top_countries = top_countries_data.groupby(['Country','City']).size().sort_values(ascending=False).reset_index(name='Number of Vehicles')
#visualize the top 10 cities
top_cities = ev_city_dist_top_countries.head(10)
plt.figure(figsize=(12,8))
sns.barplot(x='Number of Vehicles', y='City', hue='Country', data=top_cities, palette='magma')
plt.title('Top cities in top countries by EV Registrations')
plt.xlabel('Number of Vehicles Registered')
plt.ylabel('City')
plt.legend(title='Country')
plt.tight_layout()
plt.show()


# # EV Types distribution

# In[12]:


ev_type_dist = df['Electric Vehicle Type'].value_counts()
plt.figure(figsize=(12,8))
sns.barplot(x=ev_type_dist.values, y= ev_type_dist.index, palette='rocket')
plt.title('Distribution of ELectric Vehicle Types')
plt.xlabel('Number of vehicles registered')
plt.ylabel('Electric Vehicle Type')
plt.tight_layout()
plt.show()


# # Make and Model distribution

# In[13]:


# Top 10 EV manufactureres
ev_make_dist =df['Make'].value_counts().head(10)
plt.figure(figsize=(12,8))
sns.barplot(x= ev_make_dist.values, y = ev_make_dist.index, palette='cubehelix')
plt.title('Top 10 popular EV makers')
plt.xlabel('Number of vehicles registered')
plt.ylabel('Make')
plt.tight_layout()
plt.show()
print('Conclusion: From the graph is clear that TESLA is marketleader in EV manufacturing followed by NISSAN & CHEVEROLET')


# In[14]:


top_3_makers =ev_make_dist.head(3).index
top_3_maker_data = df[df['Make'].isin(top_3_makers)]
ev_model_dist_top_3_maker = top_3_maker_data.groupby(['Make', 'Model']).size().sort_values(ascending=False).reset_index(name='Number of Vehicles')
top_10_models = ev_model_dist_top_3_maker.head(10)

plt.figure(figsize=(12,6))
sns.barplot(x='Number of Vehicles', y='Model', hue='Make', data=top_10_models, palette='viridis')
plt.title('Top 10 Models in top 3 makers by EV registrations')
plt.xlabel('Number of Vehicle Registered')
plt.ylabel('Model')
plt.legend(title='Make', loc='center right')
#plt.tight_layout()
plt.show


# # Electric Range distribution

# In[15]:


plt.figure(figsize=(12,8))
sns.histplot(df['Electric Range'], bins=30, kde=True, color='royalblue')
plt.title('Distribution of Electric Vehicle Ranges')
plt.xlabel('Electric Range(miles)')
plt.ylabel('Number of Vehicles')
plt.axvline(df['Electric Range'].mean(), color='red', linestyle='--', label=f'Mean Range: {df["Electric Range"].mean():.2f} miles')
plt.legend()
plt.show()


# # Average electric range by model year

# In[16]:


average_range_by_year = df.groupby('Model Year')['Electric Range'].mean().reset_index()
plt.figure(figsize=(12,6))
sns.lineplot(x='Model Year', y='Electric Range', data= average_range_by_year, marker='o')
plt.title('Average Electric Range by Model Year')
plt.xlabel('Model Year')
plt.ylabel('Average Electric Range(miles)')
plt.grid(True)
plt.show()


# In[17]:


average_range_by_model = top_3_maker_data.groupby(['Make','Model'])['Electric Range'].mean().sort_values(ascending=False).reset_index()
#taking out top 10 models by average electric range
top_range_models = average_range_by_model.head(10)
plt.figure(figsize=(12,8))
sns.barplot(x='Electric Range', y='Model', hue='Make', data=top_range_models, palette='cool')
plt.title('Top 10 Models by Average Electric Rnage in Top 3 Makers')
plt.xlabel('Average Electric range(miles)')
plt.ylabel('Model')
plt.legend(title='Make', loc='center right')
plt.show()


# # Estimated Market Size Analysis of Electric Vehicles in the United States

# In[18]:


from scipy.optimize import curve_fit
import numpy as np


# In[19]:


# calculating no of EV registered each year
ev_register_counts = df['Model Year'].value_counts().sort_index()
ev_register_counts

# filter the dataset to include years with complete data, assuming 2023 is the last complete year
filtered_years = ev_register_counts[ev_register_counts.index <= 2023]

# define a function for exponential growth to fit the data
def exp_growth(x,a,b):
    return a*np.exp(b*x)

# Prepare the data for curve fitting
x_data = filtered_years.index - filtered_years.index.min()
y_data = filtered_years.values

# fit the data to the exponential growth function
params, covariance = curve_fit(exp_growth, x_data, y_data)

#use the fitted function to forecast the no of EV for years 2024 to 2030
forecast_years = np.arange(2024, 2030) - filtered_years.index.min()
forecasted_values = exp_growth(forecast_years,*params)

# create a dictionary to display the forecasted values
forecasted_evs = dict(zip(forecast_years + filtered_years.index.min(), np.round(forecasted_values, 0)))
forecasted_evs


# # Plotting the estimated market size data:

# In[20]:


years = np.arange(filtered_years.index.min(), 2030)
actual_years = filtered_years.index
forecast_years = np.arange(2024, 2030)
actual_values = filtered_years.values
forecasted_values

plt.figure(figsize=(12,8))
plt.plot(actual_years, actual_values,'bo-', label='Actual Registrations')
plt.plot(forecast_years, forecasted_values, 'ro-', label='Forecasted Registrations')
plt.title('Current & Estimated EV Market')
plt.xlabel('Year')
plt.ylabel('Number of EV Registrations')
plt.legend()
plt.grid(True)
plt.show


# In[ ]:





# In[ ]:




