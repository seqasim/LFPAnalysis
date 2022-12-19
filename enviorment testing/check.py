# Open the file in read mode
with open('environment2.yml', 'r') as f:
  # Read all lines in the file
  lines = f.readlines()

# Open the file in write mode
with open('environment2.yml', 'w') as f:
  # Loop through each line
  for line in lines:
    # Find the last occurrence of "=" in the line
    last_equal_index = line.rfind("=")
    # If "=S" was found and the line has exactly two "=", delete all text after and including the last "="
    if last_equal_index != -1 and line.count("=") == 2:
      line = line[:last_equal_index]
    # Write the modified line to the file
    f.write(line)