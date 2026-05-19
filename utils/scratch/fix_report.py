
import sys

filepath = 'report.md'
with open(filepath, 'r') as f:
    content = f.read()

content = content.replace('- Added generated images per row: `1000`\\n\\n| Ratio | Setting | Active components | Test ACC | Best Val ACC | Test loss | Log |', 
                          '- Added generated images per row: `1000`\n\n| Ratio | Setting | Active components | Test ACC | Best Val ACC | Test loss | Log |')

with open(filepath, 'w') as f:
    f.write(content)
