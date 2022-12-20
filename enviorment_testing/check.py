#prompt
#go to directory
#conda env create environment
# https://python-bloggers.com/2021/09/creating-and-replicating-an-anaconda-environment-from-a-yaml-file/#:~:text=Creating%20and%20replicating%20an%20Anaconda%20Environment%20from%20a,to%20replicate%20environment%20...%205%20To%20close%E2%80%A6%20


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
    if last_equal_index != -1 and line.count("=") == 1:
      line = line[:last_equal_index]
    # Write the modified line to the file
    f.write(line)