import json

def concatenate_defect_energies_summary(defect_energy_summary_1, defect_energy_summary_2, save_output_path):
    # Load the JSON data from the two files
    with open(defect_energy_summary_1, 'r') as f1:
        data1 = json.load(f1)

    with open(defect_energy_summary_2, 'r') as f2:
        data2 = json.load(f2)

    # Merge "defect_energies"
    defect_energies1 = data1.get("defect_energies", {})
    defect_energies2 = data2.get("defect_energies", {})

    merged_defect_energies = defect_energies1.copy()  # Start with the first file's data

    for key, value in defect_energies2.items():
        if key not in merged_defect_energies:
            merged_defect_energies[key] = value  # Add the new key-value pair

    # Merge "rel_chem_pots"
    rel_chem_pots1 = data1.get("rel_chem_pots", {})
    rel_chem_pots2 = data2.get("rel_chem_pots", {})

    merged_rel_chem_pots = {}

    # Merge section "A"
    merged_rel_chem_pots['A'] = {}
    for key, value in rel_chem_pots1.get('A', {}).items():
        merged_rel_chem_pots['A'][key] = value  # Add all names from the first file
    for key, value in rel_chem_pots2.get('A', {}).items():
        if key not in merged_rel_chem_pots['A']:
            merged_rel_chem_pots['A'][key] = value  # Add new names from the second file

    # Merge section "B"
    merged_rel_chem_pots['B'] = {}
    for key, value in rel_chem_pots1.get('B', {}).items():
        merged_rel_chem_pots['B'][key] = value  # Add all names from the first file
    for key, value in rel_chem_pots2.get('B', {}).items():
        if key not in merged_rel_chem_pots['B']:
            merged_rel_chem_pots['B'][key] = value  # Add new names from the second file

    # Prepare the final data structure
    merged_data = data1.copy()  # Start with the first file's data
    merged_data["defect_energies"] = merged_defect_energies  # Replace defect energies
    merged_data["rel_chem_pots"] = merged_rel_chem_pots  # Replace rel chem pots

    # Write the merged data to the output file
    with open(save_output_path, 'w') as out_file:
        json.dump(merged_data, out_file, indent=4)