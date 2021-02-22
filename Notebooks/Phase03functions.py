from operator import itemgetter
def get_object_columns_unique_count(df):
    """
    Takes in a dataframe and returns the column names and number of unique values for object columns in a list of dictionaries. Serves to see the number of unique categorical values in the dataset and hopefully allow us to bin correspondingly.
    ========================
    inputs
    ========================
    df: Pandas dataframe
    
    ========================
    outputs
    ========================
    sorted_obj_cols: A list of dictionaries containing the column name and the number of unique items in each dictionary. The dictionary is sorted by column with the highest number of unique values to column with the least number of unique values.
    """
    object_cols = []
    for column in df.columns:
    
        if df[column].dtype == 'object':
            object_cols.append({'column' : column, 'unique items':len(df[column].unique())})
        else:
            continue
    sorted_obj_cols = sorted(object_cols, key=itemgetter('unique items'), reverse=True)
    
    return sorted_obj_cols



def show_status_group_values(df, category, subcategory, show_normalized_values = False):
    """
    This function will take two categories of hierarchical values, display the parent and child values of said categories
    and then give the status group values of the parent and child values. This relates to columns that relate to one another.

    For example, 'quality_group' as the parent column and 'water_quality' as the child column within this dataset. For all rows that
    have 'salty' as their value in the 'quality_group' column, that same row will either have 'salty' or 'salty abandoned' as
    their value in the 'water_quality' column.

    This function prints the 'status_group' value counts for each parent and child in order to determine at a glance if the child 
    category is redundant. If there is too little in the counts of a child or if the normalized values are too similar, then
    the child column is redundant and will be dropped.
    ===========
    inputs
    ===========
    df: a pandas DataFrame
    
    category: str. column name within df
    
    subcategory: str. column name within df. Very important that this column relates to category in some way. This column will be a more granular version of what was submitted in 'category' and the two should share a hierarchical relationship
    
    show_normalize values: bool {True, False} default is False. Instead of displaying the value counts numbers, it will display them as percentages. Use counts to show whether the column is too granular to be used and use the normalized counts to know if the percentage difference in the status group is significant.
    
    ==========
    outputs
    ==========
    None
    """

    for item in df[category].unique():
        print('\n\n', "{} Column value: ".format(category) + " {}".format(item))
        keys = df[df[category] == item][subcategory].value_counts().keys()
        print(keys)
        for key in keys:
            print("\n"+"{} value of ".format(key) + "{} column".format(subcategory))
            print(df[df[subcategory] == key]['status_group'].value_counts(normalize = show_normalized_values))
            
    return



def group_into_contribution_size(df, column, group_size_bounds = (500, 1000), name = 'Medium sized contributor', name_2 = 'Small sized contributor'):
    """
    This function will take in a dataframe, a column, and will rename the values within the dataset
    in accordance to their contribution size, this is in effort to reduce the number of unique categorical values in the installer
    and funder columns of this dataset.
    ==========
    inputs
    ==========
    df: pandas DataFrame
    
    column: str, column name within df, should be a categorical column
    
    group_size_bounds: tuple, int. These numbers represent the lower and upper bounds for the total counts for each unique value to be considered for grouping. If a unique value has a count higher than the upper bound, it will keep it's uniqueness. If it is within the bounds, it will be reassigned a new name. If you wish to create only one new group, assign lower bound to 0.
    
    name: str. new value name for all unique values whose counts fall in between the group_size_bounds
    
    name_2: str. new value name for all unique values whose counts fall beneath the lower bound of group_size_bounds
    
    ==========
    outputs
    ==========
    returns a printed statement of the new value counts for the designated column.
    
    CAUTION: All changes done by this function permanently change the dataframe. Know what you want the bounds to be before using.
    """
    new_vals = {}
    for index, count in zip(df[column].value_counts().index, df[column].value_counts()):
        rcount = round(count, -1)
        
        #gathers the index fi the value falls within a certain bound into a dictionary
        if rcount >= 1000:
            continue
            
        elif rcount < Med_size_bounds[1] and rcount >= Med_size_bounds[0]:
            new_vals[index] = "{}".format(name)
            
        elif rcount < Med_size_bounds[0]:
            new_vals[index] = "{}".format(name_2)
            
    df[column].replace(new_vals, inplace=True)
    return print("New unique value set \n\n {}".format(df[column].value_counts()))