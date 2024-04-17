# Gaining Insights into Group-Level Course Difficulty via Differential Course Functioning

## Inroduction

This repository contains a reference implementation for _Gaining Insights into Group-Level Course Difficulty via Differential Course Functioning_, as accepted at the ACM Conference on Learning @ Scale (L@S ’24) and for _Gaining Insights into Course Difficulty Variations Using Item Response Theory_, as accepted at the 14th International Learning Analytics and Knowledge Conference 2024. 
If you use this implementation in academic work to detect differential course function, please cite the following paper:

* Baucks\*, F., Schmucker\*, R., Borcher, C., Pardos, Z., Wiskott, L. (2024). Gaining Insights into Group-Level Course Difficulty via Differential Course Functioning. In: To be updated (Eds.). Proceedings of the 11th ACM Conference on Learning @ Scale (L@S ’24). accepted.

```bibtex
@inproceedings{BaucksSchmucker2024DCF,
    author       = {Baucks*, Frederik and Schmucker*, Robin and Borchers, Conrad and Pardos, Zachary and Wiskott, Laurenz},
    title        = {Gaining Insights into Group-Level Course Difficulty via Differential Course Functioning},
    booktitle    = {{Proceedings of the 11th ACM Conference on Learning @ Scale (L@S ’24)}},
    date         = {2024-07-18},
    year         = {2024},
    venue        = {Atlanta, GA, USA},
    editor       = {To be updated},
    note         = {accepted}
}
```

If you use this implementation to only employ item response theory and calculate course difficutlies, please cite the following paper:

* Baucks\*, F., Schmucker\*, R., Wiskott, L. (2024). Gaining Insights into Course Difficulty Variations Using Item Response Theory. In LAK24: 14th International Learning Analytics and Knowledge Conference (LAK ’24). (pp. 450–461) New York, NY, USA: Association for Computing Machinery

```bibtex
@inproceedings{BaucksSchmucker2024IRT,
	author		=	{Baucks, Frederik and Schmucker, Robin and Wiskott, Laurenz},
	title		=	{Gaining Insights into Course Difficulty Variations Using Item Response Theory},
	booktitle	=	{LAK24: 14th International Learning Analytics and Knowledge Conference},
	pages		=	{450–461},
	publisher	=	{Association for Computing Machinery},
	address		=	{New York, NY, USA},
	month		=	{March},
	year		=	{2024},
	doi		=	{10.1145/3636555.3636902},
}
```

## Formatting Student Response Data

First, a folder must be created in data/real for each degree (see e.g. 'data/real/a' or 'data/real/b' - 'a' and 'b' are artificial data). The response data must be stored in this folder in .csv format with sep=',' and filename taggedRInput.csv. The exact formatting can be seen in the example degrees 'a' and 'b'. 

The responses are stored as a course-response matrix. The row names (index) correspond to the course names, the column names correspond to the student IDs. 

EXCEPTION: The first column must be called 'time' and contain the semester names. These must be coded as 'SS10' or 'WS10/11'. SS stands for summer semester, 10 for the year 2010, and WS for winter semester, 10/11 for the years 2010/2011. 

The entries in the matrix (except for the 'time' column) are then the responses of the students in the courses. Missing values are coded as -99999. 

The data must be available for single degrees only. In the case of multiple degrees, the folders and data are automatically generated from the single-degree folders.

## Quickstart

Follow these steps to set up your environment and quickly start the project:

**Clone the Repository**  
   
   To clone the repository, run the following command:
   ```bash
   git clone https://github.com/FrederikBaucks/irt-course-evaluation.git
   cd irt-course-evaluation

   ```
   To install python packages and R libraries:

   ```bash
   pip install -r config/python_packages
   Rscript -e 'packages <- scan("config/R_libraries", what="", quiet=TRUE); install.packages(packages, repos="http://cran.us.r-project.org")'

   ```
**Open notebooks/expl.ipynb**  

We share artificial data ('data/real/a' and 'data/real/b') to protect students' privacy. The artificial data has the same format and the expl.ipynb notebook illustrates the complete analysis workflow.

Run the cells in the notebook: notebooks/expl.ipynb!

When using your data: CUSTOMIZATION NEEDED in expl.ipynb! In the first cell of the notebook, the degree names need to be defined. In the case of a single degree as ['degree name', None] and in the case of multiple degrees as ['degree name 1', 'degree name 2']. In addition, the passing and failing grades must be specified. 




