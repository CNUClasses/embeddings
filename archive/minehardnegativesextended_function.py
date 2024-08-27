try:
    from sentence_transformers.util import mine_hard_negatives_extended
except Exception as e:
    from typing import Optional, Literal
    import torch
    import random
    import pandas as pd
    import datasets
    from datasets import Dataset

    #added by KP
    def dropduplicaterows(ds: "Dataset"):
        """
        Remove duplicate rows from a dataset.  Expects Dataset to have a single column

        Args:
            ds (huggingface dataset): The input dataset.

        Returns:
            datasets.Dataset: The dataset with duplicate rows removed.
        """
        # just 1 column
        columns = ds.column_names
        if len(columns) != 1:
            raise ValueError("Dataset must contain exactly two columns.")
        
        ds = pd.DataFrame(ds)
        ds = ds.drop_duplicates()
        ds = datasets.Dataset.from_pandas(ds, preserve_index=False)
        return ds
    
    def mine_hard_negatives_extended(
        dataset: "Dataset",
        model: "SentenceTransformer",
        corpus: "Dataset" = None,
        cross_encoder: Optional["CrossEncoder"] = None,
        range_min: int = 0,
        range_max: Optional[int] = None,
        max_score: Optional[float] = None,
        margin: Optional[float] = None,
        num_negatives: int = 3,
        sampling_strategy: Literal["random", "top"] = "top",
        as_triplets: bool = True,
        batch_size: int = 32,
        use_faiss: bool = False,
        verbose: bool = True,
    ) -> "Dataset":
        """
        Add hard negatives to a dataset of (anchor, positive) pairs to create (anchor, positive, negative) triplets or
        (anchor, positive, negative_1, ..., negative_n) tuples.

        Hard negative mining is a technique to improve the quality of a dataset by adding hard negatives, which are
        texts that may appear similar to the anchor, but are not. Using hard negatives can improve the performance of
        models trained on the dataset.

        This function uses a SentenceTransformer model to embed the sentences in the dataset, and then finds the closest
        matches to each anchor sentence in the dataset. It then samples negatives from the closest matches, optionally
        using a CrossEncoder model to rescore the candidates.

        You can influence the candidate negative selection in various ways:

        - **range_min**: Minimum rank of the closest matches to consider as negatives: useful to skip the most similar texts to
        avoid marking texts as negative that are actually positives.
        - **range_max**: Maximum rank of the closest matches to consider as negatives: useful to limit the number of candidates
        to sample negatives from. A lower value makes processing faster, but may result in less candidate negatives that
        satisfy the margin or max_score conditions.
        - **max_score**: Maximum score to consider as a negative: useful to skip candidates that are too similar to the anchor.
        - **margin**: Margin for hard negative mining: useful to skip candidates negatives whose similarity to the anchor is
        within a certain margin of the positive pair. A value of 0 can be used to enforce that the negative is always
        further away from the anchor than the positive.
        - **sampling_strategy**: Sampling strategy for negatives: "top" or "random". "top" will always sample the top n
        candidates as negatives, while "random" will sample n negatives randomly from the candidates that satisfy the
        margin or max_score conditions.

        Example:

            >>> from sentence_transformers.util import mine_hard_negatives
            >>> from sentence_transformers import SentenceTransformer
            >>> from datasets import load_dataset
            >>> # Load a Sentence Transformer model
            >>> model = SentenceTransformer("all-MiniLM-L6-v2")
            >>>
            >>> # Load a dataset to mine hard negatives from
            >>> dataset = load_dataset("sentence-transformers/natural-questions", split="train")
            >>> dataset
            Dataset({
                features: ['query', 'answer'],
                num_rows: 100231
            })
            >>> dataset = mine_hard_negatives(
            ...     dataset=dataset,
            ...     model=model,
            ...     range_min=10,
            ...     range_max=50,
            ...     max_score=0.8,
            ...     margin=0.1,
            ...     num_negatives=5,
            ...     sampling_strategy="random",
            ...     batch_size=128,
            ...     use_faiss=True,
            ... )
            Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 784/784 [00:43<00:00, 17.83it/s]
            Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 784/784 [00:07<00:00, 99.60it/s]
            Querying FAISS index: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 784/784 [00:00<00:00, 884.99it/s]
            Metric       Positive       Negative     Difference
            Count         100,231        431,255        431,255
            Mean           0.6866         0.4289         0.2804
            Median         0.7010         0.4193         0.2740
            Std            0.1125         0.0754         0.0999
            Min            0.0303         0.1720         0.1001
            25%            0.6221         0.3747         0.1991
            50%            0.7010         0.4193         0.2740
            75%            0.7667         0.4751         0.3530
            Max            0.9584         0.7743         0.7003
            Skipped 1289492 potential negatives (25.23%) due to the margin of 0.1.
            Skipped 39 potential negatives (0.00%) due to the maximum score of 0.8.
            Could not find enough negatives for 69900 samples (13.95%). Consider adjusting the range_max, range_min, margin and max_score parameters if you'd like to find more valid negatives.
            >>> # Note: The minimum similarity difference is 0.1001 due to our margin of 0.1
            >>> dataset
            Dataset({
                features: ['query', 'answer', 'negative'],
                num_rows: 431255
            })
            >>> dataset[0]
            {
                'query': 'when did richmond last play in a preliminary final',
                'answer': "Richmond Football Club Richmond began 2017 with 5 straight wins, a feat it had not achieved since 1995. A series of close losses hampered the Tigers throughout the middle of the season, including a 5-point loss to the Western Bulldogs, 2-point loss to Fremantle, and a 3-point loss to the Giants. Richmond ended the season strongly with convincing victories over Fremantle and St Kilda in the final two rounds, elevating the club to 3rd on the ladder. Richmond's first final of the season against the Cats at the MCG attracted a record qualifying final crowd of 95,028; the Tigers won by 51 points. Having advanced to the first preliminary finals for the first time since 2001, Richmond defeated Greater Western Sydney by 36 points in front of a crowd of 94,258 to progress to the Grand Final against Adelaide, their first Grand Final appearance since 1982. The attendance was 100,021, the largest crowd to a grand final since 1986. The Crows led at quarter time and led by as many as 13, but the Tigers took over the game as it progressed and scored seven straight goals at one point. They eventually would win by 48 points – 16.12 (108) to Adelaide's 8.12 (60) – to end their 37-year flag drought.[22] Dustin Martin also became the first player to win a Premiership medal, the Brownlow Medal and the Norm Smith Medal in the same season, while Damien Hardwick was named AFL Coaches Association Coach of the Year. Richmond's jump from 13th to premiers also marked the biggest jump from one AFL season to the next.",
                'negative': "2018 NRL Grand Final The 2018 NRL Grand Final was the conclusive and premiership-deciding game of the 2018 National Rugby League season and was played on Sunday September 30 at Sydney's ANZ Stadium.[1] The match was contested between minor premiers the Sydney Roosters and defending premiers the Melbourne Storm. In front of a crowd of 82,688, Sydney won the match 21â€“6 to claim their 14th premiership title and their first since 2013. Roosters five-eighth Luke Keary was awarded the Clive Churchill Medal as the game's official man of the match."
            }
            >>> dataset.push_to_hub("natural-questions-hard-negatives", "triplet-all")

        Args:
            dataset (Dataset): A dataset containing (anchor, positive) pairs.
            model (SentenceTransformer): A SentenceTransformer model to use for embedding the sentences.
            corpus (Dataset): A dataset with one column, containing documents as strings that will be retrieved for negative document candidates. Defaults to None, in which case the positives will be used as corpus.
            cross_encoder (CrossEncoder, optional): A CrossEncoder model to use for rescoring the candidates. Defaults to None.
            range_min (int): Minimum rank of the closest matches to consider as negatives. Defaults to 0.
            range_max (int, optional): Maximum rank of the closest matches to consider as negatives. Defaults to None.
            max_score (float, optional): Maximum score to consider as a negative. Defaults to None.
            margin (float, optional): Margin for hard negative mining. Defaults to None.
            num_negatives (int): Number of negatives to sample. Defaults to 3.
            sampling_strategy (Literal["random", "top"]): Sampling strategy for negatives: "top" or "random". Defaults to "top".
            as_triplets (bool): If True, returns up to `num_negatives` (anchor, positive, negative) triplets for each input sample.
                If False, returns 1 (anchor, positive, negative_1, ..., negative_n) tuple for each input sample. Defaults to True.
            batch_size (int): Batch size for processing. Defaults to 32.
            use_faiss (bool): Whether to use FAISS for similarity search. May be recommended for large datasets. Defaults to False.
            verbose (bool): Whether to print statistics and logging. Defaults to True.

        Returns:
            Dataset: A dataset containing (anchor, positive, negative) triplets or (anchor, positive, negative_1, ..., negative_n) tuples.
        """
        # if not is_datasets_available():
        #    raise ImportError("Please install `datasets` to use this function: `pip install datasets`.")

        from datasets import Dataset

        if range_max is None:
            if margin is not None or max_score is not None:
                # 1 positive, 10 * num_negatives negatives because some might be skipped, and range_min skipped
                range_max = range_min + (num_negatives * 10) + 1
            else:
                # 1 positive, num_negatives negatives, and range_min skipped
                range_max = range_min + num_negatives + 1
            if verbose:
                print(f"Setting range_max to {range_max} based on other parameters.")

        # Combine anchor and positive sentences to get unique corpus
        columns = dataset.column_names
        if len(columns) != 2:
            raise ValueError("Dataset must contain exactly two columns.")

        log_counters = {}
        queries = dataset[columns[0]]
        positives = dataset[columns[1]]

        if corpus is None:
            corpus = positives
        else:
            corpus_columns = corpus.column_names
            if len(corpus_columns) != 1:
                raise ValueError("Corpus must contain exactly one column.")
            corpus=dropduplicaterows(corpus)
            corpus = corpus[corpus_columns[0]]
            
        print(f"Corpus length: {len(corpus)}, number unique in corpus= {list(set(corpus)).__len__()}")
        print(f"Queries length: {len(queries)}, number unique in queries= {list(set(queries)).__len__()}")

        # Embed the corpus and queries
        corpus_embeddings = model.encode(corpus, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
        query_embeddings = model.encode(queries, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
        positives_embeddings = model.encode(positives, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
        batch_idx = torch.arange(len(queries)).unsqueeze(-1)

        if use_faiss:
            import faiss

            # Compute the positive scores separate from FAISS
            positive_scores = model.similarity_pairwise(query_embeddings, positives_embeddings).cpu()

            query_embeddings = query_embeddings.cpu().numpy()
            corpus_embeddings = corpus_embeddings.cpu().numpy()
            index = faiss.IndexFlatIP(len(corpus_embeddings[0]))
            # Move the index to the GPU if available
            try:
                co = faiss.GpuMultipleClonerOptions()
                co.shard = True
                co.useFloat16 = True
                index: faiss.IndexFlatIP = faiss.index_cpu_to_all_gpus(index, co=co)
            except Exception:
                pass
            index.add(corpus_embeddings)
            scores_list = []
            indices_list = []
            # Iterate over query embeddings in batches so we can track the progress
            for i in trange(0, len(query_embeddings), batch_size, desc="Querying FAISS index"):
                query_chunk = query_embeddings[i : i + batch_size]
                scores, indices = index.search(query_chunk, k=range_max + 1)
                scores_list.append(scores)
                indices_list.append(indices)
            scores = torch.from_numpy(np.concatenate(scores_list, axis=0))
            indices = torch.from_numpy(np.concatenate(indices_list, axis=0))
        else:
            # Compute all similarity scores
            scores = model.similarity(query_embeddings, corpus_embeddings).cpu()
            positive_scores = model.similarity(
                query_embeddings, positives_embeddings
            ).cpu()
            positive_scores = positive_scores.diagonal().clone()
		

            # Keep only the range_max + 1 highest scores. We offset by 1 to potentially include the positive pair
            scores, indices = torch.topk(scores, k=range_max + 1, dim=1)
        del query_embeddings
        del corpus_embeddings

        # Scores is a [num_queries, range_max + 1] tensor, where we set the values to -inf to disqualify the corresponding
        # text as a negative candidate. Here we disqualify the positive pair

        # positive_indices = indices == torch.arange(len(queries), device=indices.device).unsqueeze(-1)
        # scores[positive_indices] = -float("inf")
        positive_indices = []
        for positive in positives:
            if positive in corpus:
                index = corpus.index(positive)
                positive_indices.append(index)
            else:
                positive_indices.append(-1)
        positive_indices = torch.Tensor(positive_indices).to(torch.int32).unsqueeze(-1)
        positive_indices = indices == positive_indices

        scores[positive_indices] = -float("inf")

        num_candidates = scores.numel()

        # Rescore with cross_encoder
        if cross_encoder is not None and (margin is not None or max_score is not None):
            for idx, candidate_neg_idx in tqdm(enumerate(indices), desc="Rescoring with CrossEncoder", total=len(indices)):
                query = queries[idx]
                candidate_passages = [corpus[neg_idx] for neg_idx in candidate_neg_idx]
                pred_scores = cross_encoder.predict(
                    list(zip([query] * (range_max + 1), candidate_passages)),
                    batch_size=batch_size,
                    convert_to_tensor=True,
                )
                # If we rescored a positive pair, make sure that it is disqualified again
                # if idx in candidate_neg_idx:
                #    pred_scores[candidate_neg_idx == idx] = -float("inf")

                if positives[idx] in candidate_passages:
                    positives_index = candidate_passages.index(positives[idx])
                    pred_scores[positives_index] = -float("inf")

                scores[idx] = pred_scores
            positive_scores = cross_encoder.predict(
                list(zip(queries, positives)),
                batch_size=batch_size,
                convert_to_tensor=True,
            )

        # Remove based on margin
        if margin is not None:
            removed_indices = scores + margin > positive_scores.repeat(scores.size(1), 1).T
            scores[removed_indices] = -float("inf")

            num_skipped = removed_indices.sum().item()
            if num_skipped:
                log_counters["margin"] = {
                    "skipped": num_skipped,
                    "ratio": num_skipped / num_candidates,
                }
                num_candidates -= num_skipped

        # Remove based on max_score
        if max_score is not None:
            removed_indices = scores > max_score
            scores[removed_indices] = -float("inf")

            num_skipped = removed_indices.sum().item()
            if num_skipped:
                log_counters["max_score"] = {
                    "skipped": num_skipped,
                    "ratio": num_skipped / num_candidates,
                }

        # Grab the top negative candidates and remove the first range_min candidates
        negative_scores, local_indices = torch.topk(scores, k=range_max, dim=1)
        indices = indices[batch_idx, local_indices]
        if range_min:
            indices = indices[:, range_min:]
            negative_scores = negative_scores[:, range_min:]

        # Either grab the top negatives or sample randomly
        if sampling_strategy == "top":
            indices = indices[:, :num_negatives]
            negative_scores = negative_scores[:, :num_negatives]
        elif sampling_strategy == "random":
            # Prevent sampling -inf values if possible
            num_options = indices.size(1) - negative_scores.isinf().sum(1)
            num_options = num_options.clamp(min=num_negatives)
            # Randomly sample negatives from each row
            sampled_idx = [random.sample(range(options), k=num_negatives) for options in num_options]
            indices = indices[batch_idx, sampled_idx]
            negative_scores = negative_scores[batch_idx, sampled_idx]
            # Resort the indices and scores
            negative_scores, local_indices = negative_scores.sort(dim=1, descending=True)
            indices = indices[batch_idx, local_indices]

        if as_triplets:
            # negative_scores is [num_queries, num_negatives], but may contain some -inf values if not enough negatives were found
            indices_to_keep = negative_scores != -float("inf")
            # This turns indices and negative_scores into 1d tensors
            indices = indices[indices_to_keep]
            negative_scores = negative_scores[indices_to_keep]
            anchor_indices = torch.arange(len(queries), device=indices_to_keep.device).repeat(num_negatives, 1).T
           
            anchor_indices = anchor_indices[indices_to_keep]

            triplets_data = {
                columns[0]: [],
                columns[1]: [],
                "negative": [],
            }
            for anchor_idx, negative_idx in zip(anchor_indices, indices):
                triplets_data[columns[0]].append(queries[anchor_idx])
                triplets_data[columns[1]].append(positives[anchor_idx])
                triplets_data["negative"].append(corpus[negative_idx])
            difference_scores = positive_scores.repeat(num_negatives, 1).T[indices_to_keep] - negative_scores

        else:
            # Keep only indices where num_negative negatives were found
            indices_to_keep = (negative_scores != -float("inf")).all(dim=1)
            negative_scores = negative_scores[indices_to_keep]
            indices = indices[indices_to_keep]

            # Create a list of (anchor, positive, negative_1, ..., negative_`num_negatives`) tuples
            triplets_data = {
                columns[0]: [queries[idx] for idx in range(len(queries)) if indices_to_keep[idx]],
                columns[1]: [positives[idx] for idx in range(len(positives)) if indices_to_keep[idx]],
                **{
                    f"negative_{i}": [corpus[neg_idx] for neg_idx in neg_indices]
                    for i, neg_indices in enumerate(indices.T, start=1)
                },
            }
            # Flatten it so we can use for logging
            negative_scores = negative_scores.flatten()
            difference_scores = positive_scores[indices_to_keep].repeat(num_negatives, 1).T.flatten() - negative_scores

            print("positive scores", positive_scores.size())
            print("negative scores", negative_scores.size())

        if len(triplets_data) == 0:
            raise ValueError("No triplets could be generated. Please check the parameters and dataset.")
        triplets_dataset = Dataset.from_dict(triplets_data)

        # Report some statistics
        if verbose:
            row_format = "{:<6} {:>14} {:>14} {:>14}"
            formatter = (
                lambda value: f"{value.item():.4f}"
                if isinstance(value, torch.Tensor)
                else f"{value:,}"
            )
            print(row_format.format("Metric", "Positive", "Negative", "Difference"))
            for metric, function in [
                ("count", len),
                ("mean", torch.mean),
                ("median", torch.median),
                ("std", torch.std),
                ("min", torch.min),
                ("25%", lambda scores: torch.quantile(scores.float(), q=0.25)),
                ("50%", lambda scores: torch.quantile(scores.float(), q=0.5)),
                ("75%", lambda scores: torch.quantile(scores.float(), q=0.75)),
                ("max", torch.max),
            ]:
                print(
                    row_format.format(
                        metric.capitalize(),
                        formatter(function(positive_scores)),
                        formatter(function(negative_scores)),
                        formatter(function(difference_scores)),
                    )
                )

            if "margin" in log_counters:
                print(
                    f"Skipped {log_counters['margin']['skipped']} potential negatives ({log_counters['margin']['ratio']:.2%}) due to the margin of {margin}."
                )
            if "max_score" in log_counters:
                print(
                    f"Skipped {log_counters['max_score']['skipped']} potential negatives ({log_counters['max_score']['ratio']:.2%}) due to the maximum score of {max_score}."
                )

            missing_negatives = (num_negatives * len(queries)) - len(negative_scores)
            if missing_negatives > 0:
                solutions = ["range_max"]
                if range_min > 0:
                    solutions.append("range_min")
                if margin is not None:
                    solutions.append("margin")
                if max_score is not None:
                    solutions.append("max_score")
                considerations = ", ".join(solutions[:-1])
                if len(solutions) > 1:
                    considerations += " and " + solutions[-1]
                missing_negatives_ratio = missing_negatives / (num_negatives * len(queries))
                print(
                    f"Could not find enough negatives for {missing_negatives} samples ({missing_negatives_ratio:.2%})."
                    f" Consider adjusting the {considerations} parameter{'s' if len(solutions) > 1 else ''} if you'd like to find more valid negatives."
                )

        return triplets_dataset