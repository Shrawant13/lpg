import base64
from io import BytesIO
import matplotlib.pyplot as plt


fig = plt.figure()

tmpfile = BytesIO()

fig.savefig(tmpfile, format='png')
encoded = base64.b64encode(tmpfile.getvalue())

html = 'Some html head' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + 'Some more html'

with open('predict.html','w') as f:
	f.write(html)