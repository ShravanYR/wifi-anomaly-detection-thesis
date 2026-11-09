import os
input_folder = "data_raw_traffic_burst"
output_file = "combined_apstats.txt"

# IMPORTANT: normal FIRST, anomaly SECOND
file_order = [
    "active_trafficB_normal1_20240702_202339.txt",
    "active_trafficB_anomaly1_20240702_203547.txt"
]

with open(output_file, "w", encoding="utf-8") as outfile:

    for filename in file_order:

        filepath = os.path.join(input_folder, filename)

        print(f"Adding: {filename}")

        with open(filepath, "r", encoding="utf-8") as infile:
            outfile.write(infile.read())
            outfile.write("\n")

print("\n✅ Files combined correctly.")

# import os

# Training data
# input_folder = "data_raw_normal"
# output_file = "combined_apstats.txt"

# files = sorted(
#     [f for f in os.listdir(input_folder)
#      if f.endswith(".txt")]
# )

# print(f"\nFound {len(files)} files.\n")

# with open(output_file, "w", encoding="utf-8") as outfile:

#     for i, filename in enumerate(files, start=1):

#         filepath = os.path.join(input_folder, filename)

#         print(f"[{i}/{len(files)}] Adding: {filename}")

#         with open(filepath, "r", encoding="utf-8") as infile:
#             outfile.write(infile.read())
#             outfile.write("\n")

# print("\n✅ Combined successfully:")
# print(output_file)

# import os

# # ============================================================
# # Combine AP Stats Training Files
# # WiFi Thesis
# # ============================================================

# # Folder containing all raw AP statistics files
# INPUT_FOLDER = "data_raw_active"

# # Combined output filename
# OUTPUT_FILE = "combined_apstats.txt"


# def main():

#     if not os.path.exists(INPUT_FOLDER):
#         print(f"ERROR: Folder '{INPUT_FOLDER}' does not exist.")
#         return

#     # Sort files alphabetically.
#     # Since filenames contain YYYYMMDD_HHMMSS,
#     # this is also chronological order.
#     files = sorted(
#         [
#             f for f in os.listdir(INPUT_FOLDER)
#             if f.lower().endswith(".txt")
#         ]
#     )

#     if len(files) == 0:
#         print("No .txt files found.")
#         return

#     print("=" * 60)
#     print("WiFi Thesis - AP Stats File Combiner")
#     print("=" * 60)

#     print(f"\nInput Folder : {INPUT_FOLDER}")
#     print(f"Files Found  : {len(files)}")
#     print(f"Output File  : {OUTPUT_FILE}\n")

#     with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

#         for i, filename in enumerate(files, start=1):

#             filepath = os.path.join(INPUT_FOLDER, filename)

#             print(f"[{i:03d}/{len(files):03d}] {filename}")

#             with open(filepath, "r", encoding="utf-8") as infile:
#                 outfile.write(infile.read())

#             # Separate files with one blank line
#             outfile.write("\n")

#     size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)

#     print("\n" + "=" * 60)
#     print("COMBINATION COMPLETE")
#     print("=" * 60)
#     print(f"Output File : {OUTPUT_FILE}")
#     print(f"Files Added : {len(files)}")
#     print(f"Size        : {size_mb:.2f} MB")
#     print("=" * 60)


# if __name__ == "__main__":
#     main()