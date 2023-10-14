import datetime

# Sample date-time strings
date_time_str = "2023-10-14 15:30:00"

# Convert the date-time string to a datetime object
date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')

# Extract numerical features
year = date_time_obj.year
month = date_time_obj.month
day = date_time_obj.day


# Print the extracted features
print("Year:", year)
print("Month:", month)
print("Day:", day)


