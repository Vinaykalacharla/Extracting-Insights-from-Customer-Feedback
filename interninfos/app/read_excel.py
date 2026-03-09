
import openpyxl

# Load the workbook
wb = openpyxl.load_workbook('userreviews.xlsx')

# Select the active sheet
sheet = wb.active

# Get column names (assuming first row is headers)
columns = [cell.value for cell in sheet[1]]

print("Columns:", columns)

# Print first 5 rows
print("First 5 rows:")
for row in sheet.iter_rows(min_row=2, max_row=6, values_only=True):
    print(row)