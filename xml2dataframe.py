import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from datetime import datetime

class XML2DataFrame:
    """ XML2DataFrame parser
    """

    def __init__(self, path, filename):
        """ XML files are tree structures
        That's why we needed to find the root of our data
        """
        self.tree = ET.parse(path + filename)
        self.root = self.tree.getroot()
        self.root = self.root.getchildren()[-1]
        self.root = self.root.getchildren()[0]
        # In our case, the data sits behind these nodes called 'instances'
        self.instances = self.root.getchildren()[5:]
        

    def to_df(self):
        """ This method parses the XML data into a DataFrame Object
        """
        rows = []
        for instance in self.instances:
            i_children = instance.getchildren()
            row = []
            for value in i_children:
                data = value.getchildren()[0].text
                row.append(data)
            
            # row[-1] = datetime.strptime(str(datetime.today().date()) + " " + row[-1], '%Y-%m-%d %H:%M:%S:%f')
            
            row[-1] = datetime.strptime(row[-1], '%H:%M:%S:%f').time()
            rows.append(row)
            # print(row)
        df = pd.DataFrame(rows, columns=['x', 'y', 't'])
        
        df['x'] = df['x'].astype('float32')
        df['y'] = df['y'].astype('float32')
        return df

if __name__ == '__main__':
    xml2df = XML2DataFrame('../trajectories/2d/', '009_2.xml')
    df = xml2df.to_df()
    print(type(df['x'][0]))
    print(type(df['y'][0]))
    print(type(df['t'][0]))
    print(df)