import os

path = '/Users/dimitrygrebenyuk/Yandex.Disk.localized/Working/PDF/Refinements/PDF-Cluster-Prediction/all_data/cifs/new/'

# Iterate through all CIF files in the specified path
for cif_file_name in os.listdir(path):
    if cif_file_name.endswith('.cif'):
        cif_file_path = os.path.join(path, cif_file_name)

        # Read the CIF file line by line
        with open(cif_file_path, 'r') as file:
            lines = file.readlines()

        # Split lines into separate structures and write to individual CIF files
        current_structure_lines = []
        for line in lines:
            if line.startswith('data_'):
                if current_structure_lines:
                    refcode = current_structure_lines[0].strip().split('_')[-1]  # Extracting just the refcode (last part)
                    output_cif_path = path + refcode + '.cif'
                    with open(output_cif_path, 'w') as output_file:
                        output_file.write('\n'.join(current_structure_lines))
                    current_structure_lines = []
            current_structure_lines.append(line.strip())
        if current_structure_lines:
            refcode = current_structure_lines[0].strip().split('_')[-1]  # Extracting just the refcode (last part)
            output_cif_path = path + refcode + '.cif'
            with open(output_cif_path, 'w') as output_file:
                output_file.write('\n'.join(current_structure_lines))
