import os
from openpyxl import Workbook

filenames = ["static", "ri", "po", "ph", "pl", "p", "vo", "mv", "mtbv", "af", "up"]

# Basisordner (kannst du anpassen)
base_folder = "D:/Datastream/US/"

# 47 Ordner erstellen (01 bis 31)
for i in range(1, 32):
    folder_name = f"{i:02d}"  # Format mit führender Null
    folder_path = os.path.join(base_folder, folder_name)
    os.makedirs(folder_path, exist_ok=False)

    # In jedem Ordner Excel-Dateien erstellen
    for name in filenames:
        file_name = f"{name}_{folder_name}.xlsx"
        file_path = os.path.join(folder_path, file_name)
        wb = Workbook()
        wb.save(file_path)

print("All files and folders have been generated.")