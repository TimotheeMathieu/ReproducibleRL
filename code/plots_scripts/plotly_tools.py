import plotly.graph_objects as go
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px

pad_left = 0
share_coloraxes = False

class plotly_figs():
    header  = """
    <head>
    <style>
    /* The sidebar menu */
.sidenav {
  height: 100%; /* Full-height: remove this if you want "auto" height */
  width: 180px; /* Set the width of the sidebar */
  position: fixed; /* Fixed Sidebar (stay in place on scroll) */
  z-index: 1; /* Stay on top */
  top: 0; /* Stay at the top */
  left: 0;
  background-color: #111; /* Black */
  overflow-x: hidden; /* Disable horizontal scroll */
  padding-top: 20px;
}

/* The navigation menu links */
.sidenav a {
  padding: 6px 8px 6px 8px;
  text-decoration: none;
  font-size: 18px;
  color: #818181;
  display: block;
}


.sidenav ul {
    margin-block-start: 0.5em;
    margin-block-end: 0.5em;
    padding-top: 0;
    margin-top: 0em;
    margin-inline-start: 0px;
    margin-inline-end: 0px;
    padding-inline-start: 0.5em;
} 
/* When you mouse over the navigation links, change their color */
.sidenav a:hover {
  color: #f1f1f1;
}

/* Style page content */
.main {
  margin-left: 200px; /* Same as the width of the sidebar */
  padding: 0px 10px;
}

/* On smaller screens, where height is less than 450px, change the style of the sidebar (less padding and a smaller font size) */
@media screen and (max-height: 450px) {
  .sidenav {padding-top: 15px;}
  .sidenav a {font-size: 18px;}
} 
    </style>
    </head>
    <body>
    """
    def __init__(self, include_plotlyjs='cdn',offline=False, auto_open=False):
        self.html_text = ""
        self.include_plotlyjs = include_plotlyjs
        self.title_current = 0
        self.level = 0
        self.curr_show = 0
        self.is_first_export = True
        self.toc = '<div class="sidenav">\n'
        
    def add_title(self, title):
        if self.title_current == 0:
            self.toc += f"<a href=#title_{self.title_current}>{title}</a><ul>"
        else:
            self.toc += f"</ul><a href=#title_{self.title_current}>{title}</a><ul>"
        self.html_text += f"<h1 id=title_{self.title_current}>{title}</h1>"
        self.level = 0
        self.title_current += 1

    def add_fig(self, fig, title):
        self.html_text += f"<h2 id={self.title_current}_{self.level}></h2>"
        self.toc += f"<a href=#{self.title_current}_{self.level}>{title}</a>"
        self.level += 1
        self.html_text += self.fig_to_html(fig)

    def fig_to_html(self, fig):
        if self.is_first_export:
            res = fig.to_html(include_plotlyjs=self.include_plotlyjs)
            self.is_first_export = False
        else:
            res = fig.to_html(full_html=False, include_plotlyjs=False)
        return res

    def add_menu_figs(self, parameter, title, figs_dic):

        self.html_text += f"<h2 id={self.title_current}_{self.level}></h2><div>"
        self.toc += f"<a href=#{self.title_current}_{self.level}>{title}</a>s"
        self.level += 1

        titles = list(figs_dic.keys())
        elements = [ f'var e_{f} = document.getElementById("idShowMe_{self.curr_show + f}");' for f in range(len(titles))]
        switch = []
        for i in range(len(titles)):
            txt = ""
            t = titles[i]
            txt += f'case "{t}":\n'
            for f in range(len(titles)):
                if f == i:
                    txt += f'e_{f}.style.display = "block";\n'
                else:
                    txt += f'e_{f}.style.display = "none";\n'
            txt += "break;\n"
            switch.append(txt)
        
        self.html_text += '''
    <script>
    function showMe'''+str(self.curr_show)+'''(e) {
        var strdisplay = e.options[e.selectedIndex].value;
        '''+("\n".join(elements))+'''
        switch (strdisplay){
        '''+("\n".join(switch))+'''
        }
    }
    </script>
        '''
        self.html_text += parameter+': <select onchange="showMe'+str(self.curr_show)+'(this);">\n'
        self.html_text += '\n'.join(["<option>"+t+"</option>" for t in titles])
        self.html_text+= "</select>\n "

        self.html_text += f'<div id="idShowMe_{self.curr_show}" style="display: block">\n<b>'
        print(titles)
        self.html_text += self.fig_to_html(figs_dic[titles[0]])
        self.html_text += '</b></div>\n'

        for i in range(1,len(titles)):
            self.html_text += f'<div id="idShowMe_{self.curr_show +i}" style="display: none">\n<b>'
            self.html_text += self.fig_to_html(figs_dic[titles[i]])
            self.html_text += '</b></div>\n'

        self.html_text += '</div>'
        self.curr_show += len(titles)
    def save_to_file(self, fname):
        with open(fname, 'w') as f:
            f.write('<div class="main"> '+ self.header)
            f.write(self.toc+"</ul>"+"</div>")
            f.write(self.html_text)
            f.write('</div> ')
