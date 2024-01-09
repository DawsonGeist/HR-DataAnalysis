import pandas as pd
import numpy as np

filepath = "DataSets/HRdata.csv"

def GetDataFrame():
    # df is short for dataframe
    df = pd.read_csv(filepath)
    return df

def ViewColumnUniqueValues(data, column):
    uniqueVals = data.loc[:, column].unique()
    print(uniqueVals)

def TestIfTwoColumnsAreDirectlyCorrelated(data, column1Label, column2Label):
    isValid = True
    # Pair the unique values in col1 to the unique values in col2
    colValPair = {}

    col1 = data.loc[:, column1Label]
    col2 = data.loc[:, column2Label]

    # print("col1 Unique:")
    col1U = col1.unique()
    # print(col1U)
    # print("col2 Unique:")
    col2U = col2.unique()
    # print(col2U)

    # If the # of unique values in col1 != # of unique values in col2 they can't be directly correlated
    if(len(col1U) == len(col2U)):
        # Pair the unique values in col1 to the unique values in col2
        colValPair = {}
        for index, value in enumerate(col1):
            # If we currently dont have a pairing recorded for the current col1 value
            if value not in list(colValPair.keys()):
                # Add the pair into our colValPair Dictionary such that currentCol1Val : currentCol2Val
                colValPair[value] = col2[index]
            # If we do currently have a pairing recorded for the current col1 value, check if it's corresponding col2 Value is ==
            # To the currentCol2Value. If true, continue, else, break
            else:
                if(colValPair[value] != col2[index]):
                    isValid = False
                    break
        
    return (isValid, colValPair)

def ImportAndCleanData():
    # Load Data
    print("-----------------------\n\tLoad Data")
    data = GetDataFrame()
    print("-----------------------\n")



    # Drop unneccessary columns -2 0
    print("\tDrop Columns: -2, 0\n")
    data = data.drop(columns=['-2', '0'])
    print(data.columns)
    print("-----------------------\n")


    # Encode Attrition to Binary Values
    print("\tEncode Attrition to Binary Values\n")
    column = 'Attrition'
    # ViewColumnUniqueValues(data, column)
    encodedAttrition = {
        "No": 0,
        "Yes": 1,
    }
    data = data.replace({column: encodedAttrition})
    print(data.loc[:, column])
    print("-----------------------\n")


    # Test if Attrition Column Values are directly correlated to CF_attrition label
    testDf = pd.DataFrame({
        'A': [0,1,0,1],
        'B': ['a', 'b', 'a', 'a']
    })
    # print(TestIfTwoColumnsAreDirectlyCorrelated(testDf, 'A', 'B'))
    attritionLabelRelation = TestIfTwoColumnsAreDirectlyCorrelated(data, "Attrition", "CF_attrition label")
    if attritionLabelRelation[0]:
        print("Attrition - CF_attrition label relationship:")
        print(attritionLabelRelation[1])
        print("\n\tDropping CF_attrition label\n")
        data = data.drop(columns=["CF_attrition label"])
        print(data.columns)
        print("-----------------------\n")


    # Encode Business Travel to numeric Values
    print("\tEncode Business Travel to numeric Values\n")
    column = 'Business Travel'
    # ViewColumnUniqueValues(data, column)
    encodedBizTravel = {
        "Non-Travel": 0,
        "Travel_Rarely": 1,
        "Travel_Frequently": 2
    }
    data = data.replace({column: encodedBizTravel})
    print(data.loc[:,"Business Travel"])
    print("-----------------------\n")


    # Test if emp no and Employee Number are directly correlated
    empNoRelation = TestIfTwoColumnsAreDirectlyCorrelated(data, "emp no", "Employee Number")
    if empNoRelation[0]:
        print("\nemp no - Employee Number Relation:")
        print(empNoRelation[1])
        print("\n\tDropping emp no column")
        data = data.drop(columns=["emp no"])
        print(data.columns)
        print("-----------------------\n")


    # Encode CF_age band to Number Values
    print("\tEncode CF_age band to Number Values\n")
    column = 'CF_age band'
    ViewColumnUniqueValues(data, column)
    encodedAgeBand = {
        "Under 25": 0,
        "25 - 34": 1,
        "35 - 44": 2,
        "45 - 54": 3,
        "Over 55": 0,
    }
    data = data.replace({column: encodedAgeBand})
    print(data.loc[:, column])
    print("-----------------------\n")


    # Encode Department to Number Values
    print("\tEncode Department to Number Values\n")
    column = 'Department'
    ViewColumnUniqueValues(data, column)
    encodedDepartment = {
        "Sales": 0,
        "R&D": 1,
        "HR": 2,
    }
    data = data.replace({column: encodedDepartment})
    print(data.loc[:, column])
    print("-----------------------\n")


    # Encode Education Field to Number Values
    print("\tEncode Education Field to Number Values\n")
    column = 'Education Field'
    ViewColumnUniqueValues(data, column)
    encodedEducationField = {
        "Life Sciences": 0,
        "Medical": 1,
        "Marketing": 2,
        "Technical Degree": 3,
        "Human Resources": 4,
        "Other": 5,
    }
    data = data.replace({column: encodedEducationField})
    print(data.loc[:, column])
    print("-----------------------\n")


    # Encode Gender to Number Values
    print("\tEncode Gender to Number Values\n")
    column = 'Gender'
    ViewColumnUniqueValues(data, column)
    encodedGender = {
        "Male": 0,
        "Female": 1,
    }
    data = data.replace({column: encodedGender})
    print(data.loc[:, column])
    print("-----------------------\n")


    # Encode Job Role to Number Values
    print("\tEncode Job Role to Number Values\n")
    column = 'Job Role'
    ViewColumnUniqueValues(data, column)
    encodedJobRole = {
        'Sales Executive': 0,
        'Research Scientist': 1,
        'Laboratory Technician': 2,
        'Manufacturing Director': 3,
        'Healthcare Representative': 4,
        'Manager': 5,
        'Sales Representative': 6,
        'Research Director': 7,
        'Human Resources': 8
    }
    data = data.replace({column: encodedJobRole})
    print(data.loc[:, column])
    print("-----------------------\n")


    # Encode Marital Status to Number Values
    print("\tEncode Marital Status to Number Values\n")
    column = 'Marital Status'
    ViewColumnUniqueValues(data, column)
    encodedMaritalStatus = {
        'Single': 0,
        'Married': 1,
        'Divorced': 2,
    }
    data = data.replace({column: encodedMaritalStatus})
    print(data.loc[:, column])
    print("-----------------------\n")


    # Encode Over time to Number Values
    print("\tEncode Over Time to Number Values\n")
    column = 'Over Time'
    ViewColumnUniqueValues(data, column)
    encodedOverTime = {
        'No': 0,
        'Yes': 1,
    }
    data = data.replace({column: encodedOverTime})
    print(data.loc[:, column])
    print("-----------------------\n")


    # Encode Over18 to Number Values
    print("\tEncode Over18 to Number Values\n")
    column = 'Over18'
    ViewColumnUniqueValues(data, column)
    #encodedOver18 = {
    #    'N': 0,
    #    'Y': 1,
    #}
    # data = data.replace({column: encodedOver18})
    # print(data.loc[:, column])

    print("\nOnly 1 Unique value for column... Dropping column")
    data = data.drop(columns=['Over18'])
    print(data.columns)

    print("-----------------------\n")


    # Encode CF_current Employee to Number Values
    print("\tEncode CF_current Employee to Number Values\n")
    column = 'CF_current Employee'
    currentEmployeeAttritionRelation = TestIfTwoColumnsAreDirectlyCorrelated(data, "Attrition", column)
    if currentEmployeeAttritionRelation[0]:
        print("CF_current Employee Directly Correlated to Attrition")
        print("Attrition - CF_current Employee Relation:")
        print(currentEmployeeAttritionRelation[1])
        print("Dropping CF_current Employee")
        data = data.drop(columns=[column])
        print("-----------------------\n")


    # Encode Education to Number Values
    print("\tEncode Education to Number Values\n")
    column = 'Education'
    ViewColumnUniqueValues(data, column)
    encodedEducation = {
        'High School': 0,
        'Associates Degree': 1,
        "Bachelor's Degree": 2,
        "Master's Degree": 3,
        "Doctoral Degree": 4,
    }
    data = data.replace({column: encodedEducation})
    print(data.loc[:, column])
    print("-----------------------\n")


    # Encode Satisfaction to Number Values
    print("\tEncode Environment Satisfaction to Number Values\n")
    column = 'Environment Satisfaction'
    ViewColumnUniqueValues(data, column)
    encodedEducation = {
        'High School': 0,
        'Associates Degree': 1,
        "Bachelor's Degree": 2,
        "Master's Degree": 3,
        "Doctoral Degree": 4,
    }
    #data = data.replace({column: encodedEducation})
    print(data.loc[:, column])
    print("-----------------------\n")

    print(data.loc[0,:])

    dataEncodingMap = {
        "Attrition" : encodedAttrition,
        "AgeBand" : encodedAgeBand,
        "BusinessTravel" : encodedBizTravel,
        "Department" : encodedDepartment,
        "Education" : encodedEducation,
        "EducationField" : encodedEducationField,
        "Gender" : encodedGender,
        "JobRole" : encodedJobRole,
        "MaritalStatus" : encodedMaritalStatus,
        "OverTime" : encodedOverTime
    }
    return (data, dataEncodingMap)



