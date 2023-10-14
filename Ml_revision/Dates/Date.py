# play with date formate
import datetime

# Define a function to parse and standardize the date format
def parse_date(date_str):
    try:
        # Attempt to parse the date using the first format (MM/DD/YYYY)
        date_obj = datetime.datetime.strptime(date_str, '%m/%d/%Y')
    except ValueError:
        # If the first format fails, try the second format (M/D/YYYY)
        date_obj = datetime.datetime.strptime(date_str, '%m/%d/%Y')
    return date_obj

# Example usage:
date1_str = "12/31/2006"
date2_str = "1/3/1954"

date1_obj = parse_date(date1_str)
date2_obj = parse_date(date2_str)

print(date1_obj)
print(type(date2_obj))
