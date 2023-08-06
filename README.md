# EquitiesPairTrading
The code in this repo downloads data from yahoo finance and saves it locally in a MongoDB.
It then creates a pair trading dashboard for you to explore relationships between pairs of equities and backtest strategies.



## Set-Up

This tool requires MongoDB to be set-up locally.
The community edition can be installed on Windows by following the instructions here: https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-windows/#install-mongodb-community-edition

This tool also has dependencies on other python libraries defined in the requirements.txt file and has been tested on python 3.9. Please ensure these are installed before running.

Once you have installed the relevant dependencies, you can run the data_loader.py file to download data into a local MongoDB.
This will delete any existing data in the collection and download full history.
The functions I created in data_loader.py also allow for updating an existing collection with new data, so please use this for subsequent updates instead of running the data_loader script.

Once you have downloaded the relevant data, you can run the app.py file.
This creates a plotly dash which can be viewed on port 8051 (http://localhost:8051/).
If you run this from the command line, you should be shown a link to click to access the dash in your browser.

Follow the instructions on the dash to select a pair, review the automated analysis, backtest a trading strategy and evaluate the performance of the strategy.