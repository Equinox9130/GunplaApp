import matplotlib
matplotlib.use('Agg')

import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Load the Gunpla data
gunpla = pd.read_csv("gunplaCSV_Graded.csv", encoding="ISO-8859-1")

searchArray = []
for i in range(len(gunpla)):
    gName = gunpla.iloc[i, 0]
    searchArray.append(gName)

# Preprocessing the data
gunpla["Release_Year"] = gunpla["Release Date"].str[:4].astype("Int64")

# One-hot magic making the large similarity table
gunplaEncoded = pd.get_dummies(gunpla, columns=["Continuity", "Grade", "First appearance"])

# Scaling the Release Year so it doesn't dominate similarity comparisons
scaler = MinMaxScaler()
gunplaEncoded["Release_Year_Scaled"] = scaler.fit_transform(gunplaEncoded[["Release_Year"]])

# Dropping columns that aren't compared including the non-scaled release year
features = gunplaEncoded.drop(columns=["Product Name", "Release Date", "MSRP", "Release_Year"])

# Similarity calculation
simDF = pd.DataFrame(cosine_similarity(features), index=gunpla['Product Name'], columns=gunpla['Product Name'])

# An array to hold items to be presented in the UI
userKits = []

# A cleansed array for the similarity calculation
userCompare = []


# Function to get recommendations
def get_recommendations(userCompare):
    avg_similarity = simDF[userCompare].mean(axis=1)
    recommendations = avg_similarity.drop(labels=userCompare).sort_values(ascending=False).head(5)
    return recommendations


# Function to increase intuitive handling of adding kits
def searchKits(query, selected_grade):
    # Search in Product Name
    filtered = gunpla[gunpla["Product Name"].str.contains(query, case=False, na=False, regex=False)]

    # Apply Grade filter if it's set
    if selected_grade:
        filtered = filtered[filtered["Grade"] == selected_grade]

    return filtered["Product Name"].tolist()

@app.route("/search")
def search():
    query = request.args.get("q", "")
    grade = request.args.get("g", "")
    matches = searchKits(query, grade)
    return jsonify(matches)

# Backend function to clear the user's collection
@app.route("/clear", methods=["POST"])
def clear():
    userKits.clear()
    return render_template("index.html", userKits=[], recommendations=None)

# Main calculation function
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = None

    if request.method == "POST":
        builtInput = request.form['userCompare']
        userKits.append(builtInput)

        userCompare = ([name.strip() for name in userKits if name.strip() in simDF.columns])

        if userCompare:

            avgSimilarity = simDF[userCompare].mean(axis=1)
            recommended_series = avgSimilarity.drop(labels=userCompare).sort_values(ascending=False).head(5)
            top100_series = avgSimilarity.drop(labels=userCompare).sort_values(ascending=False).head(100)
            recommendations = []

            # Initialize variables for pie chart visuals
            labelsG = ["High Grades", "Master Grades", "Real Grades", "Perfect Grades", "SDs", "Others"]
            countHG = 0
            countMG = 0
            countRG = 0
            countPG = 0
            countSD = 0
            countOther = 0

            for kitName in top100_series.index:
                grade = gunpla.loc[gunpla["Product Name"] == kitName, "Grade"].values[0]
                if grade == "HG":
                    countHG += 1
                elif grade == "MG":
                    countMG += 1
                elif grade == "PG":
                    countPG += 1
                elif grade == "RG":
                    countRG += 1
                elif grade == "SD":
                    countSD += 1
                else:
                    countOther += 1

            labelsU = ["UC", "G", "00", "Wing", "SEED", "IBO", "WfM", "Other"]
            countUC = 0
            countG = 0
            count00 = 0
            countWing = 0
            countSEED = 0
            countIBO = 0
            countWfM = 0
            countElse = 0

            for kitName in top100_series.index:
                universe = gunpla.loc[gunpla["Product Name"] == kitName, "Continuity"].values[0]
                if universe == "UC":
                    countUC += 1
                elif universe == "FC":
                    countG += 1
                elif universe == "AD":
                    count00 += 1
                elif universe == "AC":
                    countWing += 1
                elif universe == "CE":
                    countSEED += 1
                elif universe == "PD":
                    countIBO += 1
                elif universe == "AS":
                    countWfM += 1
                else:
                    countElse += 1

        for kitName, score in recommended_series.items():
            recommendations.append((kitName, score))

        # Grade chart
        gradeArray = [countHG, countMG, countRG, countPG, countSD, countOther]
        filtered_grades = [(label, count) for label, count in zip(labelsG, gradeArray) if count > 0]
        if filtered_grades:
            labelsG_filtered, gradeArray_filtered = zip(*filtered_grades)
            fig1, ax1 = plt.subplots()
            ax1.pie(gradeArray_filtered, labels=labelsG_filtered, autopct='%1.1f%%', shadow=True, startangle=90)
            plt.savefig('static/gradePie.png')
            plt.close()

        # Universe chart
        universeArray = [countUC, countG, count00, countWing, countSEED, countIBO, countWfM, countElse]
        filtered_universes = [(label, count) for label, count in zip(labelsU, universeArray) if count > 0]
        if filtered_universes:
            labelsU_filtered, universeArray_filtered = zip(*filtered_universes)
            fig2, ax2 = plt.subplots()
            ax2.pie(universeArray_filtered, labels=labelsU_filtered, autopct='%1.1f%%', shadow=True, startangle=90)
            plt.savefig('static/universePie.png')
            plt.close()



    return render_template("index.html", userKits=userKits, recommendations=recommendations, )

if __name__ == "__main__":
    app.run(debug=False)