from torch import nn


class LANISTRMultiModalForPreTraining(nn.Module):
    """LANISTR class for pre-training."""

    def __init__(
            self,
            args: omegaconf.DictConfig,
            image_encoder: nn.Module,
            mim_head: nn.Module,
            text_encoder: nn.Module,
            mlm_head: nn.Module,
            tabular_encoder: nn.Module,
            timeseries_encoder: nn.Module,
            mm_fusion: nn.Module,
            image_proj: nn.Module,
            text_proj: nn.Module,
            tabular_proj: nn.Module,
            time_proj: nn.Module,
            mm_proj: nn.Module,
            mm_predictor: nn.Module,
    ):
        super().__init__()

        self.mlm_probability = args.mlm_probability
        self.args = args

        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.tabular_encoder = tabular_encoder
        self.timeseries_encoder = timeseries_encoder
        self.mm_fusion = mm_fusion

        self.image_proj = image_proj
        self.text_proj = text_proj
        self.tabular_proj = tabular_proj
        self.time_proj = time_proj

        self.mm_predictor = mm_predictor
        self.mm_proj = mm_proj

        self.mmm_loss = NegativeCosineSimilarityLoss
        self.target_token_idx = 0

        self.mlm_head = mlm_head(text_encoder.config)
        self.mlm_loss_fcn = nn.CrossEntropyLoss()  # -100 index = padding token

        self.image_encoder.embeddings.mask_token = nn.Parameter(
            torch.zeros(1, 1, image_encoder.config.hidden_size)
        )
        self.mim_head = mim_head(image_encoder.config)

        self.mtm_loss_fcn = MaskedMSELoss(reduction='none')

    def forward(
            self, batch: Mapping[str, torch.Tensor]
    ) -> LANISTRMultiModalForPreTrainingOutput:
        """Forward pass of the model.

        Args:
          batch: batch of data

        Returns:
          LANISTRMultiModalForPreTrainingOutput
        """
        loss_mlm = torch.zeros(1).to(self.args.device)
        loss_mim = torch.zeros(1).to(self.args.device)
        loss_mtm = torch.zeros(1).to(self.args.device)
        loss_mfm = torch.zeros(1).to(self.args.device)

        loss = torch.zeros(1).to(self.args.device)

        embeds = []
        masked_embeds = []

        ## ========================= MLM ================================##
        if self.args.text:
            batch['input_ids'] = batch['input_ids'].squeeze(1)
            batch['attention_mask'] = batch['attention_mask'].squeeze(1)

            # Preparing inputs and labels for MLM
            batch_size = batch['input_ids'].shape[0]
            input_ids = batch['input_ids'].clone()
            mlm_labels = input_ids.clone()
            # create random array of floats with equal dimensions to input_ids tensor
            rand = torch.rand(input_ids.shape).to(self.args.device)

            # create mask array
            mask_arr = (
                    (rand < self.mlm_probability) *
                    (input_ids != 101) *
                    (input_ids != 102) *
                    (input_ids != 0)
            )
            mask_arr = mask_arr.to(self.args.device)

            selection = [
                torch.flatten(mask_arr[i].nonzero()).tolist()
                for i in range(batch_size)
            ]

            # Then apply these indices to each respective row in input_ids, assigning
            # each of the values at these indices as 103.
            for i in range(batch_size):
                input_ids[i, selection[i]] = 103

            # input ids are now ready to be fed into the MLM encoder
            mlm_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=batch['attention_mask'],
                return_dict=True,
            )

            mlm_prediction_scores = self.mlm_head(mlm_outputs[0])
            loss_mlm = self.mlm_loss_fcn(
                mlm_prediction_scores.view(-1, self.text_encoder.config.vocab_size),
                mlm_labels.view(-1),
            )
            loss_mlm *= self.args.lambda_mlm
            loss += loss_mlm

            # Masked features and embeddings
            mlm_last_hidden_states = mlm_outputs.last_hidden_state
            mlm_text_embeddings = self.text_proj(
                mlm_last_hidden_states[:, self.target_token_idx, :]
            )
            mlm_text_embeddings = F.normalize(mlm_text_embeddings, dim=1)
            masked_embeds.append(mlm_text_embeddings.unsqueeze(dim=1))

            # forwarding non_masked inputs:
            outputs = self.text_encoder(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            last_hidden_state = outputs.last_hidden_state
            text_embeddings = self.text_proj(
                last_hidden_state[:, self.target_token_idx, :]
            )

            text_embeddings = F.normalize(text_embeddings, dim=1)
            embeds.append(text_embeddings.unsqueeze(dim=1))

        ## ============================= MIM =====================================##
        if self.args.image:
            pixel_values = batch['pixel_values'].clone()
            bool_masked_pos = batch['bool_masked_pos']
            mim_output = self.image_encoder(
                pixel_values=pixel_values, bool_masked_pos=bool_masked_pos
            )
            sequence_output = mim_output[0]
            # Reshape to (batch_size, num_channels, height, width)
            sequence_output = sequence_output[:, 1:]
            batch_size, sequence_length, num_channels = sequence_output.shape
            height = width = math.floor(sequence_length ** 0.5)
            sequence_output = sequence_output.permute(0, 2, 1).reshape(
                batch_size, num_channels, height, width
            )
            # Reconstruct pixel values
            reconstructed_pixel_values = self.mim_head(sequence_output)

            size = (
                    self.image_encoder.config.image_size //
                    self.image_encoder.config.patch_size
            )
            bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
            mask = (
                bool_masked_pos.repeat_interleave(
                    self.image_encoder.config.patch_size, 1
                )
                .repeat_interleave(self.image_encoder.config.patch_size, 2)
                .unsqueeze(1)
                .contiguous()
            )
            reconstruction_loss = nn.functional.l1_loss(
                pixel_values, reconstructed_pixel_values, reduction='none'
            )
            loss_mim = (
                    (reconstruction_loss * mask).sum() /
                    (mask.sum() + 1e-5) /
                    self.image_encoder.config.num_channels
            )
            loss_mim *= self.args.lambda_mim
            loss += loss_mim

            mim_embeddings = self.image_proj(mim_output.last_hidden_state)
            mim_embeddings = F.normalize(mim_embeddings, dim=1)
            masked_embeds.append(mim_embeddings)

            image_features = self.image_encoder(
                pixel_values=batch['pixel_values'], bool_masked_pos=None
            )
            image_embeddings = self.image_proj(image_features.last_hidden_state)
            image_embeddings = F.normalize(image_embeddings, dim=1)
            embeds.append(image_embeddings)

        ## =============================== MFM ===================================##
        if self.args.tab:
            tabular_output = self.tabular_encoder(batch['features'])
            loss_mfm = tabular_output.masked_loss
            masked_tabular_embeddings = self.tabular_proj(
                tabular_output.masked_last_hidden_state
            )
            masked_tabular_embeddings = F.normalize(masked_tabular_embeddings, dim=1)
            masked_embeds.append(masked_tabular_embeddings.unsqueeze(dim=1))
            loss_mfm *= self.args.lambda_mfm
            loss += loss_mfm

            unmasked_tabular_embeddings = self.tabular_proj(
                tabular_output.unmasked_last_hidden_state
            )
            unmasked_tabular_embeddings = F.normalize(
                unmasked_tabular_embeddings, dim=1
            )
            embeds.append(unmasked_tabular_embeddings.unsqueeze(dim=1))

        ## ================================== MTM =================================##
        if self.args.time:
            batch_size = batch['timeseries'].shape[0]

            masks = batch['noise_mask']
            lengths = [ts.shape[0] for ts in batch['timeseries']]
            x_data = torch.zeros(
                batch_size,
                self.args.timeseries_max_seq_len,
                batch['timeseries'][0].shape[-1],
            ).to(
                self.args.device
            )  # (batch_size, padded_length, feat_dim)
            target_masks = torch.zeros_like(
                x_data, dtype=torch.bool, device=self.args.device
            )  # (batch_size, padded_length, feat_dim) masks related to objective

            for i in range(batch_size):
                end = min(lengths[i], self.args.timeseries_max_seq_len)
                x_data[i, :end, :] = batch['timeseries'][i][:end, :]
                target_masks[i, :end, :] = masks[i][:end, :]

            targets = x_data.clone()
            x_data = x_data * target_masks  # mask input
            target_masks = (
                ~target_masks
            )  # inverse logic: 0 now means ignore, 1 means predict
            masked_timeseries_features = self.timeseries_encoder(
                x_data, padding_masks=batch['padding_mask']
            )

            target_masks = target_masks * batch['padding_mask'].unsqueeze(-1)
            loss_mtm = self.mtm_loss_fcn(
                masked_timeseries_features, targets, target_masks
            )
            loss_mtm = torch.sum(loss_mtm) / len(loss_mtm)
            loss_mtm *= self.args.lambda_mtm
            loss += loss_mtm

            masked_timeseries_embeddings = self.time_proj(masked_timeseries_features)
            masked_timeseries_embeddings = F.normalize(
                masked_timeseries_embeddings, dim=1
            )
            masked_embeds.append(masked_timeseries_embeddings)

            # forwarding non_masked inputs:
            timeseries_features = self.timeseries_encoder(
                batch['timeseries'], padding_masks=batch['padding_mask']
            )
            timeseries_embeddings = self.time_proj(timeseries_features)
            timeseries_embeddings = F.normalize(timeseries_embeddings, dim=1)
            embeds.append(timeseries_embeddings)

        ## ============================ MMM =====================================##
        concat_embedding = torch.cat(embeds, dim=1)
        concat_masked_embedding = torch.cat(masked_embeds, dim=1)

        mm_out = self.mm_fusion(concat_embedding)
        mm_out = mm_out.last_hidden_state

        mm_out_masked = self.mm_fusion(concat_masked_embedding)
        mm_out_masked = mm_out_masked.last_hidden_state

        z1, z2 = self.mm_proj(mm_out), self.mm_proj(mm_out_masked)
        p1, p2 = self.mm_predictor(z1), self.mm_predictor(z2)

        loss_mmm = self.mmm_loss(p1, z2) / 2 + self.mmm_loss(p2, z1) / 2
        loss += loss_mmm

        return LANISTRMultiModalForPreTrainingOutput(
            logits=p1,
            loss=loss,
            loss_mlm=loss_mlm,
            loss_mim=loss_mim,
            loss_mtm=loss_mtm,
            loss_mfm=loss_mfm,
            loss_mmm=loss_mmm,
        )


class LANISTRMultiModalModel(nn.Module):
    """LANISTR class for model's outputs."""

    def __init__(
            self,
            args: omegaconf.DictConfig,
            image_encoder: nn.Module,
            text_encoder: nn.Module,
            tabular_encoder: nn.Module,
            timeseries_encoder: nn.Module,
            mm_fusion: nn.Module,
            image_proj: nn.Module,
            text_proj: nn.Module,
            tabular_proj: nn.Module,
            time_proj: nn.Module,
            classifier: nn.Module,
    ):
        super().__init__()

        self.args = args

        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.tabular_encoder = tabular_encoder
        self.timeseries_encoder = timeseries_encoder
        self.mm_fusion = mm_fusion

        self.image_proj = image_proj
        self.text_proj = text_proj
        self.tabular_proj = tabular_proj
        self.time_proj = time_proj

        self.classifier = classifier

        self.target_token_idx = 0

    def forward(self, batch: Mapping[str, torch.Tensor]) -> BaseModelOutput:

        embeds = []
        ## ================================= Text =================================##
        if self.args.text:
            # batch['input_ids'] has shape (batch_size text_num, id_length), e.g. [4, 2, 512].
            batch_size = batch['input_ids'].shape[0]
            text_num = batch['input_ids'].shape[1]
            text_contents = batch['input_ids'].flatten(start_dim=0, end_dim=1)
            attention_mask = batch['attention_mask'].flatten(start_dim=0, end_dim=1)

            text_encoding = self.text_encoder(
                input_ids=text_contents,
                attention_mask=attention_mask,
            )
            last_hidden_state = text_encoding.last_hidden_state
            text_embeddings = self.text_proj(
                last_hidden_state[:, self.target_token_idx, :]
            )
            text_embeddings = text_embeddings.reshape(tuple([batch_size, text_num] + list(text_embeddings.shape)[1:]))

            # Average the embeddings for all the text inputs.
            text_embeddings = text_embeddings.mean(dim=1, keepdim=True)

            # TODO(Reviewer): the internal code doesn't have normalization. Do we need
            # this? Is the dimension correct? text_embeddings has shape (batch_size,
            # dim1, dim2)
            text_embeddings = F.normalize(text_embeddings, dim=1)
            embeds.append(text_embeddings)

        ## ================================== Image ===============================##
        if self.args.image:
            # batch['pixel_values'] has shape (batch_size, image_num, channel, width, height), e.g. [4, 2, 3, 224, 224].
            batch_size = batch['pixel_values'].shape[0]
            image_num = batch['pixel_values'].shape[1]
            images = batch['pixel_values'].flatten(start_dim=0, end_dim=1)

            image_encodings = self.image_encoder(
                pixel_values=images, bool_masked_pos=None
            )
            image_embeddings = self.image_proj(image_encodings.last_hidden_state)
            image_embeddings = image_embeddings.reshape(
                tuple([batch_size, image_num] + list(image_embeddings.shape)[1:])
            )
            image_embeddings = image_embeddings.mean(dim=1)

            # TODO(Reviewer): the internal code doesn't have normalization. Do we need
            # this? Is the dimension correct? image_embeddings has shape (batch_size,
            # dim1, dim2)
            image_embeddings = F.normalize(image_embeddings, dim=1)
            embeds.append(image_embeddings)

        ## ================================= Tabular ==============================##
        if self.args.tab:
            tabular_output = self.tabular_encoder(batch['features'])
            tabular_embeddings = self.tabular_proj(tabular_output.last_hidden_state)
            tabular_embeddings = F.normalize(tabular_embeddings, dim=1)
            embeds.append(tabular_embeddings.unsqueeze(dim=1))

        ## ==================================== Time ==============================##
        if self.timeseries_encoder:
            timeseries_features = self.timeseries_encoder(
                batch['timeseries'], padding_masks=batch['padding_mask']
            )
            timeseries_embeddings = self.time_proj(timeseries_features)
            timeseries_embeddings = F.normalize(timeseries_embeddings, dim=1)
            embeds.append(timeseries_embeddings)

        ## ================================ MMM ===================================##
        concat_embedding = torch.cat(embeds, dim=1)
        mm_out = self.mm_fusion(concat_embedding)
        mm_out = mm_out.last_hidden_state[:, 0, :]
        output = self.classifier(mm_out)

        ## ======================== Supervised loss ==============================##
        loss = F.cross_entropy(output, batch['labels'])

        return BaseModelOutput(
            logits=output,
            loss=loss,
        )
