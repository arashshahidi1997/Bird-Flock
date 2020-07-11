file = open("copy.txt", "w+")
file.writelines(["gadgad1234sfa\n", "jkadf2;dad"])
file.close()
file = open("copy.txt", "r+")
content = file.readlines()
print(content)
content[0] = "124ad"
print(content)
file.writelines(content)
file.close()
file = open("copy.txt", "r+")
content = file.readlines()
for line in content:
    print(line)
file.close()

