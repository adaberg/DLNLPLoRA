# Check visualizations of effective ranks, energy ratio, relative norm, sparsity in folder "lora_analysis_results";

singular values of weight matrix: sv = torch.linalg.svdvals(delta_w)

effective_rank: the number of singular values which is not smaller then 0.05*largest_sv

relative norm: delta_w.norm() / base_w.norm() (Frobenius norm);

sparsity: the number of values in weight matrix which are smaller than threshold (0.001 * largest value in delta_w.abs() / the number of total values in weight matrix

# See the analysis report in "lora_update_matrix_report.ipynb"

