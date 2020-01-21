import os, sys, re, shutil, warnings, string, fnmatch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import sqlite3, yaml
import io as pyio

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
        'colnames':None,
        'db':{
            'infocols':['project','sample', 'poreflids','thermocople_location_set','thermocople_locations_list']
        }
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

    def _db_convert_buffer(self, buffer, round=True, decimals = 2, integer = False):
        out = np.load(pyio.BytesIO(buffer))['v']
        if round:
            out = np.around(out, decimals = decimals)
        if integer or (decimals == 0):
            out = np.int64(out)
        return out

    def _db_fix_infovals(self, info):
        h = info.replace('u','')
        d = yaml.load(h, yaml.SafeLoader)
        d = pd.Series(d)
        return d
    
    def _db_get_identifier(self, s):
        return '"' + s.replace('"', '""') + '"'

    def _db_get_col_names(self, cursor, tablename):
        """Get column names of a table, given its name and a cursor
        (or connection) to the database.
        """
        reader=cursor.execute("SELECT * FROM {}".format(tablename))
        return [x[0] for x in reader.description]
    
    def db_get_run_locs(self, run_info = None, run_id = None):
        if run_info is None:
            run_info = self.db_get_run_info(run_id = run_id)
        locs = pd.DataFrame(run_info.iloc[:,-1][0], columns = ['port', 'depth'])
        locs = locs.merge(self.tc_locations, on = 'port', sort = False).loc[:,['port','x','y','depth']]
        locs.index = ['tc'+str(k).zfill(2) for k in range(1, 1+locs.shape[0])]
        return locs
    def _db_table_to_df(self, msg, columns = None):
        con = self.db_connect()
        c = con.cursor()
        c.execute(msg)
        data = c.fetchall()
        c.close()
        con.close()
        df = pd.DataFrame(data, columns = columns)
        return df
    
    def db_get_experiment_info(self, drop_invalid = True, run_id = None):

        infocols = ['run_id', 'project', 'sample', 'pore_fluids', 'start', 'end', 'summary',
                    'tc_locations_list']
        
        df1 = self.db_get_run_info(drop_invalid = drop_invalid, run_id = run_id)
        df2 = self.db_get_capture_info(run_id = run_id)

        df = pd.merge(df1, df2, left_on = 'id', right_on = 'run_id')
        df = df.loc[:, infocols]
        return df
    
    def db_get_run_info(self, drop_invalid = True, run_id = None):
        infocols = ['project','sample', 'poreflids','thermocople_location_set','thermocople_locations_list']

        msg = "SELECT id, infovals FROM database_run"
        if run_id is not None:
            msg = msg + " WHERE id="+str(int(run_id))
        
        df = self._db_table_to_df(msg, columns = ['id', 'infovals'])


        if drop_invalid:
            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop = True)
            df = pd.concat([df.loc[:,'id'], df.loc[:,'infovals'].apply(self._db_fix_infovals).loc[:,infocols]], axis = 1)
            df.replace(to_replace='None', value=np.nan, inplace = True)
            df.dropna(inplace=True, subset=['thermocople_locations_list'])
            df.reset_index(inplace=True, drop = True)
        
        df.columns = ['id', 'project','sample', 'pore_fluids','tc_locations_set','tc_locations_list']
        return df
    
    def db_get_capture_info(self, run_id = None):
        infocols = ['start', 'end', 'type', 'run_id', 'summary', 'comments']
        
        msg = "SELECT start, end, type, run_id, summary, comments FROM database_capture"
        if run_id is not None:
            msg = msg + " WHERE run_id="+str(int(run_id))
        df = self._db_table_to_df(msg, columns = infocols)
        return df
    
    def db_get_run_data(self, run_id, round_temp = True, decimals = 0, integer = True, reset_timer = True,
                        append_xyz = True, remove_not_used = True):
        """ return the run data based on a run_id"""

        msg = "SELECT abbrev,data FROM database_trace WHERE capture_id="+str(int(run_id))

        df = self._db_table_to_df(msg, columns = ['tc', 'data'])
        df = df.iloc[:-1,:]

        trace = df.loc[:,'data'].apply(self._db_convert_buffer, 
                                       round = round_temp, decimals = decimals, 
                                       integer = integer).apply(pd.Series)
        df = pd.concat([df.loc[:,'tc'], trace], axis=1)
        df.set_index('tc', inplace = True)
        df.index = [s.lower() for s in df.index.values]

        if reset_timer:
            df.loc['time',:] = df.loc['time',:] - df.loc['time',:].min()

        if append_xyz:
            locs = self.db_get_run_locs(run_id = run_id)
            df = pd.merge(locs, df, left_index = True, right_index = True, how = 'right')
            df.iloc[:,0] = df.iloc[:,0].fillna(0.5).apply(np.int64)
            df.loc[df.index[:7].values,['x','y','depth']] = 0.5
            df.loc[df.index[7],['x','y','depth']] = [0.5, 0.5, 1.0]
            df.loc[df.index[8],['x','y','depth']] = [0.8, 0.8, 0.8]
        
        if remove_not_used:
            out_index = df.index[df.iloc[:,4:].apply(np.max, axis=1)>0]
            df = df.loc[out_index,:].copy()

        return df

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

class postprocess(object):
    
    def get_tc_index(self, df):
        tcindex = re.findall(r'tc[0-9]+',str(df.index.values))
        return tcindex
    
    def data_mix_max_mean(self,df, round = True, decimals = 1):
        tcindex = self.get_tc_index(df)
        tccols  = df.columns.values[4:]
        time = df.loc['time',tccols].values
        tdata = np.array([time, df.loc[tcindex,tccols].min().values, 
                         df.loc[tcindex,tccols].max().values, 
                         df.loc[tcindex,tccols].mean().values,
                         df.loc[tcindex,tccols].std().values])
        dfout = pd.DataFrame(tdata.T, columns = ['t','min', 'max', 'mean', 'std'])
        if round: 
            dfout.iloc[:, 3:] = dfout.iloc[:, 3:].round(decimals)

        return dfout

    def __init__(self):
        super().__init__()
        return