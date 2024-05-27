# make kullm train data(JSON)

import os
import json
import pandas as pd

def make_kullm_custom_data(excel_target: str):
    """
    this function intended for converting Excel to JSON within KULLM training
    The column names in Excel are composed of "출처" (source), "제시문" (prompt), "질문" (question), and "답변" (answer)
    an element's type in JSON is a dictionary, The dictionary is composed of "instruction", "input", "output"
    
    
    first input: org excel file path(type: str)
    second input: save json file path(type: str)
    use example : make_kullm_custom_data("구름학습데이터 정제 240228.xlsx", "data_custom_240228_3104.json")
    """
    df = pd.read_excel(excel_target) # To get column names, type print(df.columns)

    result_lst = []
    
    for i in df.index:
        tmp_dict = {}
        for column_name in df.columns: # data type check
            tmp_dict[column_name] = df[column_name][i] if isinstance(df[column_name][i], str) else ''
            
        result_lst.append({"input": "제시문\n" + tmp_dict["제시문"] + "\n\n출처\n" + tmp_dict["출처"],
                          "instruction": tmp_dict["질문"],
                           "output": tmp_dict["답변"], 
                          })
    
    # preprocessed data length check
    print(f"오리지널 데이터 갯수: {len(df)}")
    print(f"정제한 데이터 갯수: {len(result_lst)}")
        
    save_path = excel_target.replace(os.path.splitext(excel_target)[-1], ".json")
    with open(save_path, "w")as f:
        json.dump(result_lst, f, ensure_ascii = False)
        
make_kullm_custom_data("../data/kullm_custom_data_240228.xlsx")