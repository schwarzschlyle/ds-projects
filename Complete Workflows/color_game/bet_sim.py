
from PIL import Image
import cv2
import numpy as np
import pyautogui
import pytesseract
import re
import time
import pandas as pd
from datetime import datetime




# Set the path to the Tesseract executable
custom_tesseract_path = 'C:/Program Files/Tesseract-OCR/tesseract.exe'  # Update the path accordingly

# Set the custom path to Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = custom_tesseract_path





n_y = 0
n_w = 0
n_p = 0
n_b = 0
n_r = 0
n_g = 0

b = 4

counter_y = 0
counter_w = 0
counter_p = 0
counter_b = 0
counter_r = 0
counter_g = 0


probabilities = input("Initial Probabilities: ")
total_games = input("Current Total Games: ")
total_games = int(total_games)


y_occurence = int(int(probabilities[0:2])*0.01*total_games*3)
w_occurence = int(int(probabilities[2:4])*0.01*total_games*3)
p_occurence = int(int(probabilities[4:6])*0.01*total_games*3)
b_occurence = int(int(probabilities[6:8])*0.01*total_games*3)
r_occurence = int(int(probabilities[8:10])*0.01*total_games*3)
g_occurence = int(int(probabilities[10:12])*0.01*total_games*3)


top_color_coordinates = (1084, 911)
mid_color_coordinates = (1084, 934)
bot_color_coordinates = (1084, 952)


# Define color categories and their corresponding RGB values
color_categories = {
    "y": (255, 255, 0),
    "w": (255, 255, 255),
    "p": (255, 192, 203),
    "b": (0, 0, 255),
    "r": (255, 0, 0),
    "g": (0, 255, 0),
    "blank": (90, 33, 62)
}


result = ""

capital = 2000

threshold = 0.010


# Read the CSV file
df = pd.read_csv('color_game_data.csv')

# Get the latest batch value
latest_batch = df['batch'].iloc[-1]


print("Latest batch:", latest_batch)

current_batch = latest_batch + 1


def extract_numbers_from_roi(x, y, width, height):
    # Capture the screen
    screenshot = pyautogui.screenshot()
    
    # Convert the screenshot to OpenCV format
    img_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    # Define the ROI coordinates
    roi = img_cv[y:y+height, x:x+width]
    
    # Convert ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Thresholding to remove overall background
    _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Convert ROI to HSV format
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Define the lower and upper color thresholds for dark red background
    lower_dark_red = np.array([0, 0, 50])
    upper_dark_red = np.array([70, 70, 150])
    
    # Apply color-based thresholding for dark red background
    mask_dark_red = cv2.inRange(hsv_roi, lower_dark_red, upper_dark_red)
    processed_roi_dark_red = cv2.bitwise_and(roi, roi, mask=mask_dark_red)
    
    # Combine thresholded images
    combined_mask = cv2.bitwise_or(thresh, mask_dark_red)
    combined_processed_roi = cv2.bitwise_and(roi, roi, mask=combined_mask)
    
    # Convert combined processed ROI to grayscale
    gray_combined_processed_roi = cv2.cvtColor(combined_processed_roi, cv2.COLOR_BGR2GRAY)
    
    # Use Tesseract OCR to extract text from the ROI
    result = pytesseract.image_to_string(gray_combined_processed_roi, config='--psm 6')
    

    return result


# Configure the regions of interest (ROI)
timer_region = (716, 226, 60, 55)  # Format: (x, y, width, height)



# Flag to track if the code block has been executed
code_executed = False


while True:
    result = extract_numbers_from_roi(*timer_region)
    
    if 'r' not in result and 't' not in result and not code_executed:
        
        # Get the screenshot of the entire screen
        screenshot = pyautogui.screenshot()


        # Get the color value at a specific point on the screen
        pixel_color = screenshot.getpixel((mid_color_coordinates))


        # Find the closest matching color category
        closest_color = None
        min_distance = float('inf')

        for category, rgb in color_categories.items():
            distance = sum(abs(c1 - c2) for c1, c2 in zip(pixel_color, rgb))
            if distance < min_distance:
                min_distance = distance
                closest_color = category


        if closest_color == "blank":
            print("No relevant data yet.")
            continue
        
        
        
        print("It's betting time!")
        print("New Balance: " + str(capital))
        
        
        total_games = total_games + 1
        
        print("Total games: " + str(total_games))


        # Get the screenshot of the entire screen
        screenshot = pyautogui.screenshot()


        # Get the color value at a specific point on the screen
        pixel_color = screenshot.getpixel((top_color_coordinates))


        # Find the closest matching color category
        closest_color = None
        min_distance = float('inf')

        for category, rgb in color_categories.items():
            distance = sum(abs(c1 - c2) for c1, c2 in zip(pixel_color, rgb))
            if distance < min_distance:
                min_distance = distance
                closest_color = category

        print("Top color result: {}".format(closest_color))

        result = result + closest_color


        # Get the color value at a specific point on the screen
        pixel_color = screenshot.getpixel((mid_color_coordinates))


        # Find the closest matching color category
        closest_color = None
        min_distance = float('inf')

        for category, rgb in color_categories.items():
            distance = sum(abs(c1 - c2) for c1, c2 in zip(pixel_color, rgb))
            if distance < min_distance:
                min_distance = distance
                closest_color = category

        print("Mid color result: {}".format(closest_color))

        result = result + closest_color


        # Get the color value at a specific point on the screen
        pixel_color = screenshot.getpixel((bot_color_coordinates))


        # Find the closest matching color category
        closest_color = None
        min_distance = float('inf')

        for category, rgb in color_categories.items():
            distance = sum(abs(c1 - c2) for c1, c2 in zip(pixel_color, rgb))
            if distance < min_distance:
                min_distance = distance
                closest_color = category

        print("Bot color result: {}".format(closest_color))

        result = result + closest_color

        print(result)



        if 'y' in result:
            multiplier = 0
            for char in result:
                if char == 'y':
                    multiplier += 1
            
            n_y = 0
            if counter_y == 1:
                capital = capital + 10
                print("It's a win")
            elif counter_y == 2:
                capital = capital
                print("Breakeven")
            elif counter_y == 3:
                capital = capital + 10
                print("It's a win")
            elif counter_y == 4:
                capital = capital + 10
                print("It's a win")
            counter_y = 0
            y_occurence = y_occurence + multiplier
        else:
            n_y += 1


            
            
            
        if 'w' in result:
            multiplier = 0
            for char in result:
                if char == 'w':
                    multiplier += 1
            
            n_w = 0
            if counter_w == 1:
                capital = capital + 10
                print("It's a win")
            elif counter_w == 2:
                capital = capital
                print("Breakeven")
            elif counter_w == 3:
                capital = capital + 10
                print("It's a win")
            elif counter_w == 4:
                capital = capital + 10
                print("It's a win")
            counter_w = 0
            w_occurence = w_occurence + multiplier
        else:
            n_w += 1

            
            

        if 'p' in result:
            multiplier = 0
            for char in result:
                if char == 'p':
                    multiplier += 1
            
            n_p = 0
            if counter_p == 1:
                capital = capital + 10
                print("It's a win")
            elif counter_p == 2:
                capital = capital
                print("Breakeven")
            elif counter_p == 3:
                capital = capital + 10
                print("It's a win")
            elif counter_p == 4:
                capital = capital + 10
                print("It's a win")
            counter_p = 0
            p_occurence = p_occurence + multiplier
        else:
            n_p += 1


            
            
        if 'b' in result:
            multiplier = 0
            for char in result:
                if char == 'b':
                    multiplier += 1
            
            n_b = 0
            if counter_b == 1:
                capital = capital + 10
                print("It's a win")
            elif counter_b == 2:
                capital = capital
                print("Breakeven")
            elif counter_b == 3:
                capital = capital + 10
                print("It's a win")
            elif counter_b == 4:
                capital = capital + 10
                print("It's a win")
            counter_b = 0
            b_occurence = b_occurence + multiplier
        else:
            n_b += 1


            
            

        if 'r' in result:
            multiplier = 0
            for char in result:
                if char == 'r':
                    multiplier += 1
            
            n_r = 0
            if counter_r == 1:
                capital = capital + 10
                print("It's a win")
            elif counter_r == 2:
                capital = capital
            elif counter_r == 3:
                capital = capital + 10
                print("It's a win")
            elif counter_r == 4:
                capital = capital + 10
                print("It's a win")
            counter_r = 0
            r_occurence = r_occurence + multiplier
        else:
            n_r += 1    


            
            
        if 'g' in result:
            multiplier = 0
            for char in result:
                if char == 'g':
                    multiplier += 1
            
            n_g = 0
            if counter_g == 1:
                capital = capital + 10
                print("It's a win")
            elif counter_g == 2:
                capital = capital
            elif counter_g == 3:
                capital = capital + 10
                print("It's a win")
            elif counter_g == 4:
                capital = capital + 10
                print("It's a win")
            counter_g = 0
            g_occurence = g_occurence + multiplier
        else:
            n_g += 1



        p_y = (y_occurence / (3*total_games))*100
        p_w = (w_occurence / (3*total_games))*100
        p_p = (p_occurence / (3*total_games))*100
        p_b = (b_occurence / (3*total_games))*100
        p_r = (r_occurence / (3*total_games))*100
        p_g = (g_occurence / (3*total_games))*100

        
        print("Updated Probabilities: ")
        print(p_y)
        print(p_w)
        print(p_p)
        print(p_b)
        print(p_r)
        print(p_g)
        
               
               
        p_y = (y_occurence / ((3*total_games+4)))*100
        p_w = (w_occurence / ((3*total_games+4)))*100
        p_p = (p_occurence / ((3*total_games+4)))*100
        p_b = (b_occurence / ((3*total_games+4)))*100
        p_r = (r_occurence / ((3* total_games+4)))*100
        p_g = (g_occurence / ((3*total_games+4)))*100
        
               
        p_y = (1-(0.01*p_y))**(3*(n_y+4))
        p_w = (1-(0.01*p_w))**(3*(n_w+4))
        p_p = (1-(0.01*p_p))**(3*(n_p+4))
        p_b = (1-(0.01*p_b))**(3*(n_b+4))
        p_r = (1-(0.01*p_r))**(3*(n_r+4))
        p_g = (1-(0.01*p_g))**(3*(n_g+4))

        
        print("Updated Projected Four-fold Probabilities: ")
               
        print(p_y)
        print(p_w)
        print(p_p)
        print(p_b)
        print(p_r)
        print(p_g)
        
        
        b_y = 0
        b_w = 0
        b_p = 0
        b_b = 0
        b_r = 0
        b_g = 0

        if p_y <= threshold:
            counter_y += 1 

        if p_w <= threshold:
            counter_w += 1 

        if p_p <= threshold:
            counter_p += 1 

        if p_b <= threshold:
            counter_b += 1 

        if p_r <= threshold:
            counter_r += 1 

        if p_g <= threshold:
            counter_g += 1 

        
        time.sleep(0.5)    
            
        
        if counter_y == 0:
            print('Yellow: 0')
            
        elif counter_y == 1:
            print('Yellow: 10')
            for i in range(0,1):
                time.sleep(0.25)
                pyautogui.click(787, 499)
            time.sleep(0.25)
            pyautogui.click(817, 562)
            
        elif counter_y == 2:
            print('Yellow: 10')
            for i in range(0,1):
                time.sleep(0.25)
                pyautogui.click(787, 499)
            time.sleep(0.25)
            pyautogui.click(817, 562)
            
        elif counter_y == 3:
            print('Yellow: 30')
            for i in range(0,3):
                time.sleep(0.25)
                pyautogui.click(787, 499)
            time.sleep(0.25)
            pyautogui.click(817, 562)
            
        elif counter_y == 4:
            print('Yellow: 60')
            for i in range(0,6):
                time.sleep(0.25)
                pyautogui.click(787, 499)
            time.sleep(0.25)
            pyautogui.click(817, 562)
            
        elif counter_y == 5:
            counter_y = 0
            capital = capital - 110
            print('Yellow: 0')
            print("Lost...")


        if counter_w == 0:
            print('White: 0')
        
        elif counter_w == 1:
            print('White: 10')
            for i in range(0,1):
                time.sleep(0.25)
                pyautogui.click(935, 505)
            time.sleep(0.25)
            pyautogui.click(975, 571)
            
        elif counter_w == 2:
            print('White: 10')
            for i in range(0,1):
                time.sleep(0.25)
                pyautogui.click(935, 505)
            time.sleep(0.25)
            pyautogui.click(975, 571)
            
        elif counter_w == 3:
            print('White: 30')
            for i in range(0,3):
                time.sleep(0.25)
                pyautogui.click(935, 505)
            time.sleep(0.25)
            pyautogui.click(975, 571)
            
        elif counter_w == 4:
            print('White: 60')
            for i in range(0,6):
                time.sleep(0.25)
                pyautogui.click(935, 505)
            time.sleep(0.25)
            pyautogui.click(975, 571)
            
        elif counter_w == 5:
            counter_w = 0
            capital = capital - 110
            print('White: 0')
            print("Lost...")
            

        if counter_p == 0:
            print('Pink: 0')
        elif counter_p == 1:
            print('Pink: 10')
            for i in range(0,1):
                time.sleep(0.25)
                pyautogui.click(1084, 489)
            time.sleep(0.25)
            pyautogui.click(1115, 556)
            
        elif counter_p == 2:
            print('Pink: 10')
            for i in range(0,1):
                time.sleep(0.25)
                pyautogui.click(1084, 489)
            time.sleep(0.25)
            pyautogui.click(1115, 556)
            
        elif counter_p == 3:
            print('Pink: 30')
            for i in range(0,3):
                time.sleep(0.25)
                pyautogui.click(1084, 489)
            time.sleep(0.25)
            pyautogui.click(1115, 556)
            
        elif counter_p == 4:
            print('Pink: 60')
            for i in range(0,6):
                time.sleep(0.25)
                pyautogui.click(1084, 489)
            time.sleep(0.25)
            pyautogui.click(1115, 556)
            
        elif counter_p == 5:
            counter_p = 0
            capital = capital - 110
            print('Pink: 0')
            print("Lost...")


        if counter_b == 0:
            print('Blue: 0')
            
            
        elif counter_b == 1:
            print('Blue: 10')
            for i in range(0,1):
                time.sleep(0.25)
                pyautogui.click(781, 633)
            time.sleep(0.25)
            pyautogui.click(825, 700)
            
        elif counter_b == 2:
            print('Blue: 10')
            for i in range(0,1):
                time.sleep(0.25)
                pyautogui.click(781, 633)
            time.sleep(0.25)
            pyautogui.click(825, 700)
            
        elif counter_b == 3:
            print('Blue: 30')
            for i in range(0,3):
                time.sleep(0.25)
                pyautogui.click(781, 633)
            time.sleep(0.25)
            pyautogui.click(825, 700)
            
        elif counter_b == 4:
            print('Blue: 60')
            for i in range(0,6):
                time.sleep(0.25)
                pyautogui.click(781, 633)
            time.sleep(0.25)
            pyautogui.click(825, 700)
            
        elif counter_y == 5:
            counter_b = 0
            capital = capital - 110
            print('Blue: 0')
            print("Lost...")


        if counter_r == 0:
            print('Red: 0')
            
        elif counter_r == 1:
            print('Red: 10')
            for i in range(0,1):
                time.sleep(0.25)
                pyautogui.click(941, 637)
            time.sleep(0.25)
            pyautogui.click(973, 688)
            
        elif counter_r == 2:
            print('Red: 10')
            for i in range(0,1):
                time.sleep(0.25)
                pyautogui.click(941, 637)
            time.sleep(0.25)
            pyautogui.click(973, 688)
            
        elif counter_r == 3:
            print('Red: 30')
            for i in range(0,3):
                time.sleep(0.25)
                pyautogui.click(941, 637)
            time.sleep(0.25)
            pyautogui.click(973, 688)
            
        elif counter_r == 4:
            print('Red: 60')
            for i in range(0,6):
                time.sleep(0.25)
                pyautogui.click(941, 637)
            time.sleep(0.25)
            pyautogui.click(973, 688)
            
        elif counter_r == 5:
            counter_r = 0
            capital = capital - 110
            print('Red: 0')
            print("Lost...")


        if counter_g == 0:
            print('Green: 0')
            
            
        elif counter_g == 1:
            print('Green: 10')
            for i in range(0,1):
                time.sleep(0.25) 
                pyautogui.click(1078, 614)
            pyautogui.click(1115, 674)
            
        elif counter_g == 2:
            print('Green: 10')
            for i in range(0,1):
                time.sleep(0.25)
                pyautogui.click(1078, 614)
            time.sleep(0.25)
            pyautogui.click(1115, 674)
            
        elif counter_g == 3:
            print('Green: 30')
            for i in range(0,3):
                time.sleep(0.25)
                pyautogui.click(1078, 614)
            time.sleep(0.25)
            pyautogui.click(1115, 674)
            
        elif counter_g == 4:
            print('Green: 60')
            for i in range(0,6):
                time.sleep(0.25)
                pyautogui.click(1078, 614)
            time.sleep(0.25)
            pyautogui.click(1115, 674)
            
        elif counter_g == 5:
            counter_g = 0
            capital = capital - 110
            print('Green: 0')
            print("Lost...")
        

        
        new_data = {
            'batch': [current_batch],
            'time': [datetime.now().time()],
            'color_1': [result[0]],
            'color_2': [result[1]],
            'color_3': [result[2]]
        }

        
        
        # Create a DataFrame from the new data
        df_new = pd.DataFrame(new_data)

        # Append the new data to the existing CSV file
        df_new.to_csv('color_game_data.csv', mode='a', header=False, index=False)

        
        code_executed = True
    
    elif 'r' in result and 't' in result and code_executed:
        
        
        print("Waiting for results...")
        code_executed = False
        
        

        
        