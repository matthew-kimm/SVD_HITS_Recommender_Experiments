all: make-targets/parameter_analysis_latex_tables.completed make-targets/parameter_analysis_heatmaps.completed make-targets/neighbor_experiments_plots.completed make-targets/global_experiment_plots.completed

make-targets/parameter_analysis.completed:
	${RECOMMENDER_EXPERIMENT_SCRIPTS_DIR}/parameter_analysis.sh

make-targets/parameter_analysis_results.completed: make-targets/parameter_analysis.completed
	${RECOMMENDER_EXPERIMENT_PYTHON_BIN} Results/results.py experiments/parameter_analysis/ "Parameter Analysis" --is_parameter_analysis
	touch make-targets/parameter_analysis_results.completed

make-targets/parameter_analysis_heatmaps.completed: make-targets/parameter_analysis_results.completed
	${RECOMMENDER_EXPERIMENT_PYTHON_BIN} Results/heatmap.py experiments/parameter_analysis/ --draw_hitsw
	${RECOMMENDER_EXPERIMENT_PYTHON_BIN} Results/heatmap.py experiments/parameter_analysis/ --draw_svd_spf
	touch make-targets/parameter_analysis_heatmaps.completed

make-targets/parameter_analysis_latex_tables.completed: make-targets/parameter_analysis_results.completed
	${RECOMMENDER_EXPERIMENT_PYTHON_BIN} Results/table_tex.py experiments/parameter_analysis/ pf
	${RECOMMENDER_EXPERIMENT_PYTHON_BIN} Results/table_tex.py experiments/parameter_analysis/ gb
	touch make-targets/parameter_analysis_latex_tables.completed

make-targets/setup_neighbor_and_global_experiments.completed: make-targets/parameter_analysis_results.completed
	${RECOMMENDER_EXPERIMENT_PYTHON_BIN} Results/setup_experiments_from_parameter_analysis.py experiments/parameter_analysis/
	touch make-targets/setup_neighbor_and_global_experiments.completed

make-targets/neighbor_experiments.completed: make-targets/setup_neighbor_and_global_experiments.completed
	${RECOMMENDER_EXPERIMENT_SCRIPTS_DIR}/neighbor_experiments.sh
	touch make-targets/neighbor_experiments.completed

make-targets/global_experiment.completed: make-targets/setup_neighbor_and_global_experiments.completed
	${RECOMMENDER_EXPERIMENT_SCRIPTS_DIR}/global_analysis.sh
	touch make-targets/global_experiment.completed

make-targets/neighbor_experiments_results.completed: make-targets/neighbor_experiments.completed
	${RECOMMENDER_EXPERIMENT_PYTHON_BIN} Results/results.py experiments/exact_grade/ "Exact Grade Match"
	${RECOMMENDER_EXPERIMENT_PYTHON_BIN} Results/results.py experiments/attribute/ "Exact Grade Match With Attribute"
	${RECOMMENDER_EXPERIMENT_PYTHON_BIN} Results/results.py experiments/letter_grade/ "Letter Grade Match"
	${RECOMMENDER_EXPERIMENT_PYTHON_BIN} Results/results.py experiments/filtered/ "Exact Grade Match With Filter"
	touch make-targets/neighbor_experiments_results.completed

make-targets/global_experiment_results.completed: make-targets/global_experiment.completed
	${RECOMMENDER_EXPERIMENT_PYTHON_BIN} Results/results.py experiments/global_analysis/ "Global Neighborhood"
	touch make-targets/global_experiment_results.completed

make-targets/neighbor_experiments_plots.completed: make-targets/neighbor_experiments_results.completed
	${RECOMMENDER_EXPERIMENT_PYTHON_BIN} Results/figures.py experiments/exact_grade/ "Exact Grade Match"
	${RECOMMENDER_EXPERIMENT_PYTHON_BIN} Results/figures.py experiments/attribute/ "Exact Grade Match With Attribute"
	${RECOMMENDER_EXPERIMENT_PYTHON_BIN} Results/figures.py experiments/letter_grade/ "Letter Grade Match"
	${RECOMMENDER_EXPERIMENT_PYTHON_BIN} Results/figures.py experiments/filtered/ "Exact Grade Match With Filter"
	touch make-targets/neighbor_experiments_plots.completed

make-targets/global_experiment_plots.completed: make-targets/global_experiment_results.completed
	${RECOMMENDER_EXPERIMENT_PYTHON_BIN} Results/figures.py experiments/global_analysis/ "Global Neighborhood" --is_global_neighborhood_analysis
	touch make-targets/global_experiment_plots.completed
