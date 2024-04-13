# Specify the path to your input CSV file
outputs = ["of_test.csv", "of_train.csv", "of_val.csv"]
# Specify the path to the output CSV file
inputs = ["mid_frame_test.csv", "mid_frame_train.csv", "mid_frame_val.csv"]

for input_csv_file, output_csv_file in zip(inputs, outputs):
    # Read the input CSV file and write modified lines to the output CSV file
    with open(input_csv_file, 'r') as input_file, open(output_csv_file, 'w') as output_file:
        for line in input_file:
            # Split each line by comma
            parts = line.strip().split(',')
            
            # Get the filename  
            filename = parts[0]
            
            # Change the file extension from png to npy
            new_filename = filename.replace('.png', '.npy')
            
            # Update the line with the new filename
            new_line = new_filename + ',' + parts[1] + '\n'
            
            # Write the modified line to the output CSV file
            output_file.write(new_line)
