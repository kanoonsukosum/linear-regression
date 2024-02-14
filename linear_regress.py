import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from sklearn.metrics import r2_score

# Import the dataset
data = pd.read_csv("top_six_economies.csv")  # load data set

# DataFrame conversion
df = pd.DataFrame(data)
# Only the entry of Japan
weeb_people = df[df['Country Name'] == 'Japan']

def find_best_linear_r_squared(data, x_column, y_columns):
    best_r_squared = -1
    best_line = None
    best_slope = None
    best_intercept = None

    for y_column in y_columns:
        # Performing linear regression
        slope, intercept, r_value, p_value, std_err = linregress(data[x_column], data[y_column])
        line = slope * data[x_column] + intercept

        # Calculate R-squared
        current_r_squared = r2_score(data[y_column], line)

        # Update best R-squared, line, slope, and intercept if the current one is better
        if current_r_squared > best_r_squared:
            best_r_squared = current_r_squared
            best_line = line
            best_slope = slope
            best_intercept = intercept

    return best_r_squared, best_line, best_slope, best_intercept

###########Part 1##############
# Plotting the filtered data using a scatter plot
plt.figure()
plt.scatter(weeb_people['Year'], weeb_people['GDP (current US$)'], label='Original Data')

# Adding labels and title
plt.xlabel('Year')
plt.ylabel('GDP (current US$)')
plt.title('Best-Fitted Linear to GDP Data with Year as X-axis')

# Find the best R-squared, line, slope, and intercept for GDP
best_r_squared_gdp, best_line_gdp, best_slope_gdp, best_intercept_gdp = find_best_linear_r_squared(weeb_people, 'Year', ['GDP (current US$)'])

# Plotting the best regression line for GDP
plt.plot(weeb_people['Year'], best_line_gdp, label=f'Best Linear Fit: R-squared={best_r_squared_gdp:.2f}, Slope={best_slope_gdp:.2e}, Intercept={best_intercept_gdp:.2e}', color='red')

# Show the legend
plt.legend()


###########Part 2##############
# Name of some long column
unemployment = 'Unemployment, total (% of total labor force) (modeled ILO estimate)'
# Plotting the filtered data using a scatter plot
plt.figure()
plt.scatter(weeb_people[unemployment], weeb_people['GDP (current US$)'], label='Original Data')

# Adding labels and title
plt.xlabel('Unemployment (% of total labor force)')
plt.ylabel("GDP (current US$)")
plt.title('Best-Fitted Linear to GDP Data with Unemployment as X-axis')

# Find the best R-squared, line, slope, and intercept for Unemployment
best_r_squared_unemployment, best_line_unemployment, best_slope_unemployment, best_intercept_unemployment = find_best_linear_r_squared(weeb_people, unemployment, ['GDP (current US$)'])

# Plotting the best regression line for Unemployment
plt.plot(weeb_people[unemployment], best_line_unemployment, label=f'Best Linear Fit: R-squared={best_r_squared_unemployment:.2f}, Slope={best_slope_unemployment:.2e}, Intercept={best_intercept_unemployment:.2e}', color='red')

# Show the legend
plt.legend()

###########Execution##############

# Example prediction for GDP with Year as the x-axis
predicted_gdp_year_2025 = best_slope_gdp * 2025 + best_intercept_gdp
print(f"Predicted GDP for the year 2025: {predicted_gdp_year_2025:.2f} (in current US$)")
predicted_gdp_year_2025 = best_slope_gdp * 2030 + best_intercept_gdp
print(f"Predicted GDP for the year 2030: {predicted_gdp_year_2025:.2f} (in current US$)")
predicted_gdp_year_2025 = best_slope_gdp * 2035 + best_intercept_gdp
print(f"Predicted GDP for the year 2035: {predicted_gdp_year_2025:.2f} (in current US$)")

# Example prediction for GDP with Unemployment as the x-axis
predicted_gdp_unemployment_5 = best_slope_unemployment * 3 + best_intercept_unemployment
print(f"Predicted GDP for an unemployment rate of 3%: {predicted_gdp_unemployment_5:.2f} (in current US$)")
predicted_gdp_unemployment_5 = best_slope_unemployment * 5 + best_intercept_unemployment
print(f"Predicted GDP for an unemployment rate of 5%: {predicted_gdp_unemployment_5:.2f} (in current US$)")
predicted_gdp_unemployment_5 = best_slope_unemployment * 7 + best_intercept_unemployment
print(f"Predicted GDP for an unemployment rate of 7%: {predicted_gdp_unemployment_5:.2f} (in current US$)")

# Show the plots
plt.show()
print("Finished!")
