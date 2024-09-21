import re

def splitting (result_text):
    pattern = re.compile(r"Restaurant: (?P<restaurant_name>.*?), Zipcode: (?P<zipcode>\d{5}), Rating: (?P<rating>\d+\.\d+), Cuisines: (?P<cuisines>.*?), Review: (?P<review>.*)")
    match = pattern.search(result_text)

    if match:
        restaurant = match.group('restaurant_name')
        zipcode = match.group('zipcode')
        rating = match.group('rating')
        cuisines = match.group('cuisines')
        review = match.group('review')
        return restaurant, zipcode, rating, cuisines, review
    else:
        return -1,-1,-1,-1,-1
    
input_string = "Restaurant: em, Zipcode: 98101, Rating: 3, Cuisines: African, American (New), Review: I like it."

Restaurant, Zipcode, Rating, Cuisines, Review = splitting(input_string)
print(Restaurant)