import gc
from math import ceil
from typing import Dict, List, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import pathos
import scipy
import sklearn as sk
from scipy.sparse import issparse
from tqdm import tqdm

# Adapted from: https://github.com/ZhuoliHuang/scPAFA/blob/main/scPAFA/_tools/_fast_UCell.py from Huang, Z., Zheng, Y., Wang, W. et al. Uncovering disease-related multicellular pathway modules on large-scale single-cell transcriptomes with scPAFA. Commun Biol 7, 1523 (2024). https://doi.org/10.1038/s42003-024-07238-7

# ref_state_genes = pd.read_csv( "<path to reference table for genes for each signature / pathway for all states>"


def fast_ucell_rank(
    adata: ad.AnnData,
    n_cores_rank: int,
    maxRank: int = 1500,
    rank_batch_size: int = 100000,
):
    """
    Perform a fast UCell analysis on single-cell RNA-seq data. Step1 ranking.
    For each cell, rank gene expression(from count or log-normalized data) from high to low.
    UCell algorithm: Andreatta, M., & Carmona, S. J. (2021). UCell: Robust and scalable single-cell gene signature scoring.

    Parameters
    ----------
    adata : ad.AnnData
        An AnnData object containing the single-cell RNA-seq data, with adata.X as raw count(or log-normalized data) is recommended.
    n_cores_rank : int
        The number of CPU cores to use for ranking.
        Please note that adjusting this parameter should take into account rank_batch_size.
        When using raw count, rank_batch_size/n_cores_rank above 10,000 is recommended. 4~8 cores when rank_batch_size is 100,000 to 200,000 works good.
        Having more CPU cores does not necessarily guarantee performance improvement.
    maxRank : int, optional
        The maximum rank for UCell. Default is 1500. maxRank must bigger than the gene numbers of the longest pathway.
    rank_batch_size : int, optional
        The batch size for ranking computation. Default is 100000.
        A smaller batch will take less memory.
        Using scaled data will cost more memory.
        Will extract a batch of rank_batch_size cells, split them into smaller chunks for multiprocessing.

    Returns
    -------
    A rank scipy sparse csr matrix with row as cells and columns as genes.

    Raises
    ------
    ValueError
        If the provided adata is not a valid AnnData object.
        If n_cores_rank is not a positive integer.
        If maxRank is not a positive integer.
        If rank_batch_size is not a positive integer.
    """

    if not isinstance(adata, ad.AnnData):
        raise ValueError("adata must be a valid AnnData object.")

    if (
        not isinstance(n_cores_rank, int)
        or n_cores_rank <= 0
        or n_cores_rank > pathos.multiprocessing.cpu_count()
    ):
        raise ValueError("n_cores_rank must be a positive integer and <= max cpu_count")

    if not isinstance(maxRank, int) or maxRank <= 0:
        raise ValueError("maxRank must be a positive integer.")

    if not isinstance(rank_batch_size, int) or rank_batch_size <= 0:
        raise ValueError("rank_batch_size must be a positive integer.")

    # a rank function based on scipy.stats.rankdata that will run on each core
    def worker(i):
        # chunk
        start = i * rank_chunk_size
        end = min((i + 1) * rank_chunk_size, len(count_batch))
        count_df = count_batch[start:end]

        # rank
        rank_array = scipy.stats.rankdata((count_df * (-1)), axis=1)
        # gene expression rank bigger than maxRank will be set to 0 to save memory
        rank_array[rank_array > maxRank] = 0
        rank_sparse_matrix = scipy.sparse.csr_matrix(rank_array)
        return rank_sparse_matrix

    # check the type of adata.X
    if issparse(adata.X):
        count = pd.DataFrame.sparse.from_spmatrix(adata.X)
        print("adata.X is csr sparse matrix")
    else:
        count = adata.X.copy()
        print("adata.X is numpy.ndarray")

    num_batches = len(count) // rank_batch_size
    if len(count) % rank_batch_size != 0:
        num_batches += 1

    num_chunks = n_cores_rank
    rank_chunk_size = ceil(rank_batch_size / num_chunks)

    print("Step1 generate rank matrix")
    print(
        str(num_batches)
        + " batches need to rank, with each max "
        + str(rank_batch_size)
        + " cells"
    )
    final_csr_matrix_list = []

    for i_batch in range(num_batches):
        print("Processing_batch_" + str((i_batch + 1)))
        start = i_batch * rank_batch_size
        end = min((i_batch + 1) * rank_batch_size, len(count))
        count_batch = count[start:end]

        csr_matrix_list = []

        with pathos.pools.ProcessPool(nodes=n_cores_rank) as pool:
            results = list(
                tqdm(
                    pool.imap(worker, range(num_chunks), chunksize=1),
                    total=num_chunks,
                    desc="Ranking Chunks",
                )
            )

        csr_matrix_list = [result for result in results]
        csr_matrix_batch = scipy.sparse.vstack(csr_matrix_list)
        final_csr_matrix_list.append(csr_matrix_batch)

    final_csr_matrix = scipy.sparse.vstack(final_csr_matrix_list)

    print("Rank done")
    print(
        "The output rank matrix can be used to calculate UCell score on different pathways or signature sets."
    )
    print(
        "The maxRank parameter use with this rank matrix in fast_ucell_score() must <= "
        + str(maxRank)
    )
    return final_csr_matrix


def fast_ucell_score(
    cell_index: Sequence,
    rankmatrix: scipy.sparse.csr_matrix,
    n_cores_score: int,
    input_dict: dict,
    maxRank: int = 1500,
    score_batch_size: int = 100000,
):
    """
    Perform a fast UCell analysis on single-cell RNA-seq data.
    UCell algorithm: Andreatta, M., & Carmona, S. J. (2021). UCell: Robust and scalable single-cell gene signature scoring.

    Parameters
    ----------
    cell_index : Sequence[str]
        The cell index of adata used in fast_ucell_rank(), for example: list(adata.obs.index).
    rankmatrix : scipy.sparse.csr_matrix
        The result of fast_ucell_rank().
    n_cores_score : int
        The number of CPU cores to use for scoring.
        For hundreds of pathways, 4~8 cores are fast enough.
        For scenarios with a high number of pathways, increasing the number of CPU cores will significantly reduce the runtime.
    input_dict : dict
        A dictionary from generate_pathway_input(), containing input data.
    maxRank : int, optional
        The maximum rank for UCell,must <= maxRank used in fast_ucell_rank().Default is 1500.
    score_batch_size : int, optional
        The batch size for scoring. Default is 100000.
        Will extract a batch of score_batch_size cells.The pathways are distributed across various cores, and calculations are performed on these cells.
    Returns
    -------
    A UCell score pandas dataframe with row index as cells and columns as pathways.

    Raises
    ------
    ValueError:
        If `cell_index` is not an array-like of strings.
        If `rankmatrix` is not a `scipy.sparse.csr_matrix`.
        If `n_cores_score` is not a positive integer or exceeds the maximum CPU count.
        If `input_dict` is not a dictionary.
        If `maxRank` is smaller than the maximum gene count among pathways or not a positive integer.
        If `score_batch_size` is not a positive integer.
    """
    if not isinstance(cell_index, Sequence):
        raise ValueError("cell_index must be a list of strings.")

    if not isinstance(rankmatrix, scipy.sparse.csr_matrix):
        raise ValueError("rankmatrix must be scipy.sparse.csr_matrix.")

    if not len(cell_index) == rankmatrix.shape[0]:
        raise ValueError(
            "The length of cell_index must match the rows number of rankmatrix."
        )

    if (
        not isinstance(n_cores_score, int)
        or n_cores_score <= 0
        or n_cores_score > pathos.multiprocessing.cpu_count()
    ):
        raise ValueError(
            "n_cores_score must be a positive integer and smaller than max cpu_count."
        )

    if not isinstance(input_dict, dict):
        raise ValueError("input_dict must be a dict.")

    if maxRank < max(input_dict["pathway_dict_length"].values()):
        raise ValueError(
            "maxRank is smaller than "
            + str(max(input_dict["pathway_dict_length"].values()))
            + " which is the genes number of the longest program."
        )
    if not isinstance(maxRank, int) or maxRank <= 0:
        raise ValueError("maxRank must be a positive integer.")

    if not isinstance(score_batch_size, int) or score_batch_size <= 0:
        raise ValueError("score_batch_size must be a positive integer.")

    # mission run for each pathway
    def process_key(key):
        length = pathway_length[key]

        num1 = (length * (length + 1)) / 2
        num2 = maxRank * length

        position = np.array(pathway_pos[key])
        subset_final_csr_matrix = final_csr_matrix_batch[:, position]

        special_index = np.array(subset_final_csr_matrix.sum(axis=1)).ravel() != 0
        overmaxRank = length - subset_final_csr_matrix.getnnz(axis=1)

        final_result = np.array(subset_final_csr_matrix.sum(axis=1)).ravel()
        final_result[special_index] = (
            final_result[special_index] + (maxRank + 1) * overmaxRank[special_index]
        )
        final_result[special_index] = 1 - (final_result[special_index] - num1) / num2
        final_result = pd.Series(final_result, name=key)
        return final_result

    print("Subset rank matrix by overlap genes in pathways")
    # final_csr_matrix = rankmatrix.copy()
    final_csr_matrix = rankmatrix[:, input_dict["intersect_position"]]

    print("Rank above maxRank to 0")
    final_csr_matrix.data[final_csr_matrix.data > maxRank] = 0
    final_csr_matrix.eliminate_zeros()

    pathway_pos = input_dict["pathway_dict_filtered_position"]
    pathway_length = input_dict["pathway_dict_length"]

    keys = list(pathway_pos.keys())

    num_batches = final_csr_matrix.shape[0] // score_batch_size
    if final_csr_matrix.shape[0] % score_batch_size != 0:
        num_batches += 1

    print("step2 calculating Score")
    print(
        str(num_batches)
        + " batches need to score, with each max "
        + str(score_batch_size)
        + " cells"
    )

    UCell_score_df = pd.DataFrame()

    for i_batch in range(num_batches):
        print("processing_batch_" + str((i_batch + 1)))
        start = i_batch * score_batch_size
        end = min((i_batch + 1) * score_batch_size, final_csr_matrix.shape[0])
        final_csr_matrix_batch = final_csr_matrix[start:end, :]

        # multi_processing
        with pathos.pools.ProcessPool(nodes=n_cores_score) as pool:
            score_results = list(
                tqdm(
                    pool.imap(
                        process_key, keys, chunksize=ceil(len(keys) / n_cores_score)
                    ),
                    total=len(keys),
                    desc="Pathways",
                )
            )

        # merge result for each batch
        Ucell_dataframe = pd.concat(score_results, axis=1)
        UCell_score_df = pd.concat(
            [UCell_score_df, Ucell_dataframe], axis=0, ignore_index=True
        )

        del score_results, Ucell_dataframe
        gc.collect()

    print("UCell done!")
    print("Returning Dataframe")
    # UCell_score_df = pd.concat(final_score_list, axis=0, ignore_index=True)
    UCell_score_df.index = cell_index

    return UCell_score_df


def map_genes_to_positions(gene_dict, gene_array):
    # a function to map gene in dict to a index of gene array
    gene_dict = gene_dict.copy()
    # Create a reverse mapping that associates gene names with their positions in the array
    gene_positions = {gene: index for index, gene in enumerate(gene_array)}

    # Update the dictionary's values to be the positions of genes in the array
    for key, value in gene_dict.items():
        if isinstance(value, list):
            # If the value is a list of genes, map all genes to their positions
            gene_dict[key] = [gene_positions[g] for g in value]
        else:
            # If the value is a single gene, map it to its position
            gene_dict[key] = gene_positions.get(value, None)

    return gene_dict


def filter_dict_by_intersection(
    input_dict: Dict[str, List[str]], target_list: List[str], min_gene_num: int
) -> Dict[str, List[str]]:
    """
    Filter a dictionary of pathways based on the minimum overlap with a target gene list.

    Parameters:
        input_dict (dict): A dictionary where keys are pathway names and values are lists of genes.
        target_list (list): A list of target genes to compare against pathway genes.
        min_gene_num (int): The minimum number of overlapping genes required to keep a pathway.

    Returns:
        dict: A filtered dictionary containing pathways with sufficient overlap with the target gene list.
    """
    filtered_dict = {}
    filtered_out_count = 0  # a_counter_for_filtered_cell

    for key, value in input_dict.items():
        intersection_count = len(set(value) & set(target_list))

        if intersection_count >= min_gene_num:
            filtered_dict[key] = value
        else:
            filtered_out_count += 1
    print(f"Filtered out {filtered_out_count} pathways")
    return filtered_dict


def generate_pathway_input(
    adata: ad.AnnData, pathway_dict: Dict[str, List[str]], min_overlap_gene: int = 3
) -> dict:
    """
    Generate pathway input data for analysis.
    Parameters
    ----------
    adata : ad.AnnData
        An AnnData object containing gene expression data.
    pathway_dict : Dict[str, List[str]]
        A dictionary where keys are pathway names and values are lists of genes.
    min_overlap_gene : int, optional
        The minimum overlap of genes required to include a pathway. Default is 3.
    Returns
    -------
    dict
        A dictionary containing the following keys and values:
        - 'pathway_dict_length' (dict): Original length of pathways.
        - 'pathway_dict_filtered_gene' (dict): Genes that overlap in adata.var_names and pathways (for fast_score_genes).
        - 'intersectgene' (array): Union set of all genes in 'pathway_dict_filtered_gene'.
        - 'intersect_position' (array): Index of 'intersectgene' in adata.var_names.
        - 'pathway_dict_filtered_position' (dict): Mapping of genes in 'pathway_dict_filtered_gene' to the index of 'intersectgene'.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("Input 'adata' must be an AnnData object.")
    if not isinstance(pathway_dict, dict):
        raise TypeError("Input 'pathway_dict' must be a dictionary.")
    if not isinstance(min_overlap_gene, int):
        raise TypeError("Input 'min_overlap_gene' must be an integer.")

    # Step1: Filter out pathways directly without affecting genes in the pathway
    pathway_dict_filtered = filter_dict_by_intersection(
        pathway_dict, target_list=list(adata.var_names), min_gene_num=min_overlap_gene
    )

    pathway_dict_length = {
        key: len(value) for key, value in pathway_dict_filtered.items()
    }

    pathway_genes = [
        item
        for value in pathway_dict_filtered.values()
        for item in (value if isinstance(value, list) else [value])
    ]
    pathway_genes = np.array(list(set(pathway_genes)), dtype="object")

    pathway_genes = pathway_genes[pd.Series(pathway_genes).isin(adata.var_names)]

    pathway_df_filtered = pd.DataFrame(
        [(k, gene) for k, v in pathway_dict_filtered.items() for gene in v],
        columns=["pathway", "gene"],
    )
    pathway_df_filtered = pathway_df_filtered.loc[
        pathway_df_filtered["gene"].isin(pathway_genes)
    ]
    filtered_gene_dict = (
        pathway_df_filtered.groupby("pathway")["gene"]
        .apply(list)[pathway_dict_filtered.keys()]
        .to_dict()
    )

    pathway_genes_position = np.where(adata.var_names.isin(pathway_genes))[0]
    pathway_genes = np.array(adata.var_names[pathway_genes_position])

    filtered_gene_dict_position = map_genes_to_positions(
        filtered_gene_dict, pathway_genes
    )

    result_dict = {
        "pathway_dict_length": pathway_dict_length,
        "pathway_dict_filtered_gene": filtered_gene_dict,
        "pathway_dict_filtered_position": filtered_gene_dict_position,
        "intersectgene": pathway_genes,
        "intersect_position": pathway_genes_position,
    }

    print(str(len(pathway_dict_filtered.keys())) + " pathways passed QC")
    print(
        "The maxRank must >= " + str(max(result_dict["pathway_dict_length"].values())),
        "(The genes number of the longest pathway)",
    )
    return result_dict


def get_state_genes(
    adata: ad.AnnData,
    reference: pd.DataFrame,  # = ref_state_genes,
    suffix: str = "",
):
    state_dict = {}
    for col in reference.columns:
        new_col = col + f"_score{suffix}"
        state_dict[new_col] = [
            g for g in reference[col].dropna().tolist() if g in adata.var.index
        ]

    return state_dict


def get_ucell_scores(
    adata: ad.AnnData,
    states_dict: dict[str : List[str]],
    genes_to_remove: dict[str : List[str]] = None,
    n_cores_rank=4,
    maxRank=1500,
    rank_batch_size=100000,
    min_overlap_gene=0,
    save_adata=True,
):
    rank_matrix = fast_ucell_rank(
        adata=adata,
        n_cores_rank=n_cores_rank,
        maxRank=maxRank,
        rank_batch_size=rank_batch_size,
    )

    states_dict_filtered = states_dict.copy()
    if genes_to_remove is not None:
        for k, v in genes_to_remove.items():
            print(states_dict_filtered[k])
            states_dict_filtered[k].remove(v)

    signature_input_dict = generate_pathway_input(
        adata=adata,
        pathway_dict=states_dict_filtered,
        min_overlap_gene=min_overlap_gene,
    )

    ucell_score_df = fast_ucell_score(
        cell_index=list(adata.obs.index),
        rankmatrix=rank_matrix,
        n_cores_score=n_cores_rank,
        input_dict=signature_input_dict,
        maxRank=maxRank,
        score_batch_size=rank_batch_size,
    )

    print(ucell_score_df.head())
    ucell_score_df.columns = [col + "_ucell" for col in ucell_score_df.columns]

    adata_new = adata.copy()
    if save_adata:
        adata_new.obs = pd.concat([adata.obs, ucell_score_df], axis=1)
    return ucell_score_df, adata_new
