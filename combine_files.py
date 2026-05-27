import os

input_folder = "data_raw_signal_degradation"
output_file = "combined_apstats.txt"

# IMPORTANT: normal FIRST, anomaly SECOND
file_order = [
    "signal_degradation_normal_20240422_162813.txt",
    "signal_degradation_away_20240422_164327.txt",
]

with open(output_file, "w", encoding="utf-8") as outfile:

    for filename in file_order:

        filepath = os.path.join(input_folder, filename)

        print(f"Adding: {filename}")

        with open(filepath, "r", encoding="utf-8") as infile:
            outfile.write(infile.read())
            outfile.write("\n")

print("\n✅ Files combined correctly.")