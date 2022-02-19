### Dataset Capabilities

The Dataset is an abstraction atop a variety of sources
   - local file system via Pandas dataframe
   - remote file systems like s3, Oracle Storage
   - advanced possibilities include using pyarrow for Apache Arrow datsets
   - interoperate with Dask.dataframes

The sdk's dataset provides the following capabilities:
- Support loading dataset from pandas, remote sources, clipboard etc
  ```
  df = pd.load_csv(...)
  ds = sdk.Dataset.fromPandas(df)
  ```
- Visualizing of dataset
    - Show all features, their data types and counts, meta data for a dataset, for example if the dataset is time-series or not.
    - Show (heatmap) of correlated features
      ```
      show_in_notebook(ds)
      ```
    - Show distribution of values for features, using a datatype-dependent
      visualizer (text vs categorical vs numerical will all render differently)
      ```
      show_in_notebook(ds, features=["col1", "col2", "col3"])
      ```

    look at http://pandas.pydata.org/pandas-docs/stable/style.html for some pandas
    examples, also seaborn https://stackoverflow.com/questions/39409866/correlation-heatmap

- Data cleaning
    - Support redaction of features
    - Handle missing values
    - Identify outliers and inconsistencies
    - Imputation
- Identify data type of features
    - Define a data type hierarchy that includes special patterns like credit cards, zip codes, phone number, etc.
- Feature selection
  - features used in a model is part of AutoML, however, for modeling that's not using AutoML the dataset
    should have feature selection capabilities using a plugin (initial should be information theoretic) - see
    http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection for examples
- Sampling
  - can be used to limit data used for modeling, can also be used to balance and unbalanced target variable.
  - "auto mode" determines if up/down is required
- Features can hold feature encoding hints
   -  these hints are part of the dataset, not the underlying pandas dataframe
- Discretization
  - of target variable for regression to classification problem

- Persistence & interchange
  - support save as snapshot (possibly parquet format), a snapshot generates a shareable URI
  - load from snapshot URI
