import os, sys, re, shutil, warnings, string, fnmatch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import sqlite3, yaml
import io as pyio
import plotly.express as px
import plotly.graph_objects as go

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

class io(object):
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
    
    def _mkdir(self, path):
        if not os.path.exists(path): os.makedirs(path)
        return
    
    def _mkdir_outpath(self, outpath=None, subdirs=None):
        if outpath is None:
            outpath = self.outpath
        
        
        self._mkdir(outpath)
        if subdirs is not None:
            for sdir in subdirs:
                self._mkdir(os.path.join(outpath,sdir))
        
        return

    def _fname_date(self, s, date_format = '%Y-%m-%d'):
        return s.strftime(date_format)

    def df_to_xyzv(self, df, itercols = None, val_ix_ini = 4, xyz_val = ['x','y','depth'], ix_ini = 1,
                   basename = None, outpath = './_out', prepend = None, header = False):
        if itercols is None:
            itercols = df.columns[val_ix_ini:]
        self._mkdir(outpath)
        for col in itercols:
            cols = xyz_val + [col]
            fname = '_'.join(s for s in [prepend, str(col).zfill(6), basename] if s) 
            #'_'.join()
            fpath = os.path.join(outpath, fname + '.csv')
            df.loc[df.index[ix_ini:],cols].to_csv(fpath, index = False, header = header)
        return

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class db_io(basics, io):
    
    _db_settings = {
        'usecols':None,
        'skiprows':None,
        'colnames':None,
        'db':{
            'infocols':['project','sample', 'poreflids','thermocople_location_set','thermocople_locations_list']
        }
    }

    probe_str = {
        'rf':'low freq*|rf|radio|low',
        'mw':'microwave|mw|micro',
        'rh':'resistance|res|heater|rh',
        'st':'steam|^st+( |:)'
    }

    exp_str = {
        'baseline':'base|baseline',
        'training':'initialization|training|train',
        'installation':'install|installation'
    }
    
    src_dict = {
        'uhffwdpower':'rf',
        'uhfamppower':'rf',
        'mwfwdpower':'mw'
    }
    exp_info = None
    
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
    
    def db_get_run_locs(self, run_id):
        run_locs_raw = self.exp_info.loc[self.exp_info['capture_id']==run_id,'tc_locations_list'].values[0]
        locs = pd.DataFrame(run_locs_raw, columns = ['port', 'depth'])
        locs = locs.merge(self.tc_locations, on = 'port', sort = False).loc[:,['port','x','y','depth']]
        locs.index = ['tc'+str(k).zfill(2) for k in range(1, 1+locs.shape[0])]
        return locs

    def _dict_to_var(self, df, var, varout, dict_str):
        for key in dict_str.keys():
            ix = df[var].apply(lambda x: re.search(dict_str[key], x) is not None)
            df.loc[ix,varout] = key
        return

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

        infocols = ['run_id', 'id_capture', 'project', 'sample', 'pore_fluids', 'start', 'end', 'summary',
                    'tc_locations_list']
        
        df1 = self.db_get_run_info(drop_invalid = drop_invalid, run_id = run_id)
        df2 = self.db_get_capture_info(run_id = run_id)

        df = pd.merge(df1, df2, left_on = 'id', right_on = 'run_id', suffixes=('_exp', '_capture'))
        df = df.loc[:, infocols]
        df.columns = [infocols[0]] + ['capture_id'] + infocols[2:]
        
        df[['project', 'sample', 'summary']] = df[['project', 'sample', 'summary']].applymap(str.lower)

        df['source'] = None
        df['source_subtype'] = None
        df['exp_type'] = 'test'

        self._dict_to_var(df, 'project', 'source', self.probe_str)
        self._dict_to_var(df, 'project', 'exp_type', self.exp_str)

        df.loc[df['source'] == 'rh','source_subtype'] = df.loc[df['source'] == 'rh','project'].apply(lambda x: re.findall(r'60|30|90', x)[0])
        df = df[['run_id','capture_id','source','source_subtype','exp_type','sample','start','end', 
                 'tc_locations_list','summary','project']]
        df.reset_index(inplace = True, drop = True)

        return df
    def db_exp_info_sanity(self):
        for row in self.exp_info.iterrows():
            db_ix  = row[0]
            db_id  = row[1]['capture_id']
            db_src = row[1]['source']


            df = self.db_get_run_data(run_id=db_id, decimals=0, remove_not_used=False)
            df = df.iloc[[1,4,6],4:].copy()
            imx = df.max(axis=1).values
            ixs = df.index[imx>0].values

            em_src_test = re.findall('uhffwdpower|mwfwdpower', ' '.join(ixs))
            if len(em_src_test)>0:
                df_src = self.src_dict[em_src_test[0]]
            else:
                if db_src is not None:
                    ph_src = re.findall('st|rh', db_src)
                    if len(ph_src)>0:
                        df_src = ph_src[0]
                    else:
                        df_src = 'rh'
                else:
                    df_src = 'rh'
            if not df_src == db_src:
                print(db_ix, db_id, db_src, df_src)
                self.exp_info.loc[db_ix,'source'] = df_src

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
        infocols = ['id', 'start', 'end', 'type', 'run_id', 'summary', 'comments']
        
        msg = "SELECT id, start, end, type, run_id, summary, comments FROM database_capture"
        if run_id is not None:
            msg = msg + " WHERE run_id="+str(int(run_id))
        df = self._db_table_to_df(msg, columns = infocols)
        df[['start','end']] = df[['start','end']].apply(pd.to_datetime, format = '%Y-%m-%d %H:%M:%S.%f')
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
            df[['x','y','depth']] = df[['x','y','depth']].apply(np.around, decimals = 4)
        
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
    

    def __init__(self, labutilspath=None, basepath = './', datadir = 'exports', outdir = '_out', dbdir = 'db', 
                 viewe_installed = False):
        if labutilspath is None:
            currpath = os.path.dirname(__file__)
            labutilspath = str(Path(currpath).parents[0])
        
        super().__init__(labutilspath=labutilspath)

        self.settings.update(self._db_settings)
        self._update_paths(basepath, datadir, outdir, dbdir)

        self.exp_info = self.db_get_experiment_info()
        if not viewe_installed:
            self.exp_info = self.exp_info.loc[self.exp_info['exp_type'] != 'installation',:]
            self.exp_info.reset_index(drop = True, inplace = True)
        return

class postprocess(db_io):

    _pp_settings = {
        'outdir_vars':['basedir','subdir','capture_id'],
        'baseoutdir_vars':['basedir','subdir']
    }

    @property
    def experiments(self):
        return self.exp_info.loc[:,['capture_id','source','source_subtype','exp_type','sample','start','end']]
    
    @property
    def _list_base_outdirs(self):
        return self.exp_dbout[self.settings['baseoutdir_vars']].astype(np.str).T.apply(lambda s: os.path.join(*s)).unique()

    @property
    def list_outdirs(self):
        return self.exp_dbout[self.settings['outdir_vars']].astype(np.str).T.apply(lambda s: os.path.join(*s))

    @property
    def list_fpaths_base(self):
        return self.exp_dbout[self.settings['outdir_vars'] + ['basename']].astype(np.str).T.apply(lambda s: os.path.join(*s))

    def get_tc_index(self, df):
        tcindex = re.findall(r'tc[0-9]+',str(df.index.values))
        return tcindex
    
    def data_min_max_mean(self,df, round = True, decimals = 1):
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
    
    def _get_ix_max(self,df2, var ='mean'):
        ixmax = df2[var] == df2[var].max()
        inmax = df2.loc[ixmax,:].index.values
        l = 0
        if len(inmax)>1:
            l = int(len(inmax)/2)
        ix = inmax[l]

        return ix

    def plot_min_max_mean(self, df2):
        fig = px.scatter()
        for var in ['min','max','mean']:
            fig.add_scatter(x=df2.t, y=df2[var], text=df2.index, name = var, 
            hovertemplate =
                "<b>%{text}</b><br><br>"
                '<b>Temp</b>:%{y} degC'+
                '<br><b>time</b>: %{x} s<br>')
        
        ix = self._get_ix_max(df2)
        xx = df2.loc[ix,'t']
        yy = 1.15*df2.loc[ix,'max']
        fig.add_shape(
                      # Line Horizontal
                      go.layout.Shape(
                          type="line",
                          x0=xx,
                          y0=00,
                          x1=xx,
                          y1=yy,
                          line=dict(
                              color="LightBlue",
                              width=4,
                              dash="dashdot",
                          ),
                  ))
        fig.update_layout(
                          xaxis ={'title':'Time (s)'},
                          yaxis ={'title':'Temperature (deg C)'})
        fig.update_layout(
                          shapes=[
                              go.layout.Shape(
                                  fillcolor="rgba(63, 81, 181, 0.2)",
                                  line={"width": 0},
                                  type="rect",
                                  x0=0,
                                  x1=xx,
                                  xref="x",
                                  y0=0,
                                  y1=0.95,
                                  yref="paper"
                              ),
                              go.layout.Shape(
                                  fillcolor="rgba(76, 175, 80, 0.1)",
                                  line={"width": 0},
                                  type="rect",
                                  x0=xx,
                                  x1=df2.iloc[-1,0],
                                  xref="x",
                                  y0=0,
                                  y1=0.95,
                                  yref="paper"
                              )
                          ]
        )
        return fig

    def _generate_exp_dbout(self):
        df = pd.DataFrame(columns=['capture_id','basedir','subdir','basename','processed'])
        df[['capture_id','basedir']] = self.experiments[['capture_id','exp_type']].copy()

        df['subdir'] = self.experiments.loc[:,['source',
                                            'source_subtype']].T.apply(lambda s: '_'.join(item for item in s if item)) 
        df['basename'] =  self.experiments.loc[:,['start',
                                                  'end']].applymap(self._fname_date).T.apply(lambda s: '_'.join(s))
        
        return df
    
    def make_outdirs(self):
        self._mkdir_outpath(subdirs = self._list_base_outdirs)

    def get_heating_cooling_cols(self, df, df2 = None, delta_t = 1800, delta_th = 0.5):
        if df2 is None:
            df2 = self.data_min_max_mean(df)
        ix = self._get_ix_max(df2)
        lframe = ix + 4
        th = np.divmod(lframe,delta_t)
        if th[1]>delta_th*delta_t:
            lframe = lframe + delta_t - th[0]
        cols_heat = np.append(df.columns.values[4:lframe:delta_t], df.columns[ix+4])
        cols_cool = np.append(df.columns[ix+4],df.columns.values[lframe::delta_t])
        return cols_heat, cols_cool
    
    def get_heating_cooling_df(self, df, df2 = None, delta_t = 1800, delta_th = 0.5):
        heat, cool = self.get_heating_cooling_frames(df = df, df2 = df2, delta_t = delta_t, delta_th = delta_th)
        df_heat = df.loc[df.index[1:],['x','y','depth'] + heat.tolist()]
        df_cool = df.loc[df.index[1:],['x','y','depth'] + cool.tolist()]
        return df_heat, df_cool

    def make_xyzval(self, run_id, decimals = 0, append_basename = True):
        df = self.db_get_run_data(run_id=run_id, decimals = decimals)
        ix = self.experiments.index[self.experiments['capture_id'] == run_id].values[0]
        basename = None
        if append_basename:
            basename = self.exp_dbout.loc[ix,'basename']
        basepath = os.path.join(self.outpath, self.list_outdirs[ix])

        heat, cool = self.get_heating_cooling_cols(df)

        self.df_to_xyzv(df, itercols = heat,
                        prepend = 'heat',
                        basename = basename, 
                        outpath = basepath)
        
        self.df_to_xyzv(df, itercols = cool,
                        prepend = 'cool', 
                        basename = basename, 
                        outpath = basepath)
        return
        #df_heat, df_cool = self.get_heating_cooling_df(df)
    def exp_info_sanity_check(self):
        self.db_exp_info_sanity()
        self.__setattr__('exp_dbout', self._generate_exp_dbout())
        return
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__setattr__('exp_dbout', self._generate_exp_dbout())
        self.settings.update(self._pp_settings)
        return