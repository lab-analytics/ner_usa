import os, sys, re, shutil, warnings, string, fnmatch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import sqlite3

class basics(object):
    _labutilspath = None
    _rock_basics_flag = True
    debug  = True
    
    _src_locations_file = None
    _tc_locations_file = None
    _nerclasspath = 'ner_usa'

    src_locations = None
    tc_locations  = None

    def _set_labutilspath(self, labutilspath):
        self.__setattr__('_labutilspath', labutilspath)
        sys.path.append(self._labutilspath)
        return
    
    def _get_labutilspath(self):
        currpath = os.path.dirname(__file__)
        labutilspath = str(Path(currpath).parents[0])
        return labutilspath

    def _load_port_locations(self, port_type):
        file = os.path.join(self._labutilspath,self._nerclasspath, port_type+'_port_locations.csv')
        locs = pd.read_csv(file)
        self.__setattr__(port_type+'_locations', locs)
        self.__setattr__('_'+port_type+'_locations_file', file)
        return

    def load_port_locations(self):
        for port in ['src', 'tc']:
            self._load_port_locations(port_type = port)
        return

    def _load_tc_locations(self):
        self._load_port_locations(port_type = 'tc')
        return
    
    def _load_src_locations(self):
        self._load_port_locations(port_type = 'src')
        return

    def plot_ports(self, port_type, figsize = (10,10), save = False, outname = 'probes_map', outpath = './'):
        try:
            df = self.__getattribute__(port_type+'_locations')
            if df is None:
                self.load_port_locations()
                df = self.__getattribute__(port_type+'_locations')
            fig, ax = plt.subplots(figsize=(10,10))
            ax.scatter(x='x',y='y', data=df)
            for i, txt in enumerate(df.loc[:,'port']):
                ax.annotate('TC'+str(txt).zfill(2), (df.loc[i,'x']+0.005, df.loc[i,'y']+0.005))
            if save:
                outfile = os.path.join(outpath, outname + '.png')
                if not os.path.exists(outpath): os.mkdir(outpath)
                plt.savefig(outfile)
        except:
            warnings.warn('there are only two types of ports:\n \t "tc": thermocouple \n \t "src":sources')
        return
    
    def __init__(self, labutilspath = None):
        if labutilspath is None:
            labutilspath = self._get_labutilspath()
        self._set_labutilspath(labutilspath)
        self.load_port_locations()
        
        return

class io(basics):
    
    settings = {
        'paths':{
            'root':None,
            'data':None,
            'out':'./_out',
            'exclude':['_special-studies', 'special_studies', '*deprecated*',
                       '_unsorted', '*layout*', '_postprocessed', '*installation*'],
            'database':'db',
            'raw':'raw'},
        'files':{
            'exclude':['.*','_*','*.asd','*.tcl', 'summary*','*map*', 'full*','combined*'],
            'include':['*.csv']
        },
        'usecols':None,
        'skiprows':None,
        'colnames':None
    }

    @property
    def datapath(self):
        return self._getpath('data')    
    @property
    def outpath(self):
        return self._getpath('out')
    @property
    def rootpath(self):
        return self._getpath('root')
    @property
    def basepath(self):
        return self.rootpath
    @property
    def dbpath(self, dbfile = 'db.sqlite'):
        return os.path.join(self._getpath('database'), dbfile)
    @property
    def _file(self):
        return os.path.dirname(__file__)
    
    def _getpath(self, value):
        return self.settings['paths'][value]

    def _update_paths(self, basepath, datadir, outdir = './_out', dbdir = 'db'):
        self.settings['paths']['root'] = basepath
        self.settings['paths']['data'] = os.path.join(basepath, datadir)
        self.settings['paths']['out']  = os.path.join(basepath, outdir)
        self.settings['paths']['database']  = os.path.join(basepath, dbdir)
        return
    
    def _list_to_regex(self, l):
        out = r'|'.join([fnmatch.translate(x) for x in l]) or r'$.'
        return out

    def _enforce_float(self, df):
        df = df.apply(pd.to_numeric,errors='coerce')
        df.replace(np.inf, np.nan, inplace = True)
        df.dropna(inplace = True)
        # reset index 
        df.reset_index(inplace = True, drop = True)

        return df
    
    def _set_outpath(self, outdir = None, mkdir = True):
        if outpath is not None:
            self.settings['path']['out'] = os.path.join(self.settings['paths']['root'], outdir)    
        outpath = self.settings['path']['out']
        if mkdir and (not os.path.exists(outpath)):
            os.mkdir(outpath)
        return outpath

    def wrangler(self):
        pass
    
    def db_connect(self):
        db_path = self.dbpath
        con = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        return con

    def __init__(self, labutilspath=None, basepath = './', datadir = 'exports', outdir = '_out', dbdir = 'db'):
        if labutilspath is None:
            currpath = os.path.dirname(__file__)
            labutilspath = str(Path(currpath).parents[0])
        super().__init__(labutilspath=labutilspath)
        self._update_paths(basepath, datadir, outdir, dbdir)
        return