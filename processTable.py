import pandas as pd

process_table = pd.DataFrame({
    "sample_id":    [1,2,3,4,5,6,7,8,9],
    "level":        [1,2,1,2,3,3,1,2,3],        # <-- fill exactly from your S.No. table
    "travel_speed": [10,10,8,8,10,8,12,12,12],   # <-- fill exactly from your S.No. table
    "heat_input":   [260.10,307.79,325.13,384.73,450.18,216.75,256.49,300.12,None]  # <-- get row 9's value, it was cut off
})
process_table.to_csv("process_table.csv", index=False)
print(process_table)
