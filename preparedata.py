import requests
import csv

# Set the GitHub repository and the API endpoint
repo_url = "https://api.github.com/repos/pytorch/pytorch"
issues_endpoint = f"{repo_url}/issues"

# Initialize an empty list to store all issues
all_issues = []

page = 1
while True:
    # Set the parameters for the API request
    params = {
        'state': 'open',  # Change to 'closed' for closed issues
        'per_page': 100,  # Number of issues per page (adjust as needed)
        'page': page
    }

    # Send the GET request to fetch issues
    response = requests.get(issues_endpoint, params=params)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        issues = response.json()
        if not issues:
            break  # No more issues to fetch
        all_issues.extend(issues)
        page += 1
    else:
        print(f"Failed to retrieve issues. Status code: {response.status_code}")
        break

# Open a CSV file for writing
with open('torch_github_issues.csv', mode='w', newline='') as csv_file:
    fieldnames = ['Serial Number', 'Issue Number', 'Title', 'Labels', 'Body','Comments']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    # Write the header row
    writer.writeheader()
    
    # Write the issues data to the CSV file
    for serial, issue in enumerate(all_issues, start=1):
        writer.writerow({
            'Serial Number': serial,
            'Issue Number': issue['number'],
            'Title': issue['title'],
            'Labels': ', '.join(label['name'] for label in issue['labels']),
            'Body': issue['body'],
            'Comments': issue['comments']
        })
    
print(f"Data written to github_issues.csv")
