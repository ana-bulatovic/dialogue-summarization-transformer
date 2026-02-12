import torch
import torch.nn as nn
import math


# Prebacuje ID tokena u kontinuirane vektore (embedding prostor)
class InputEmbeddings(nn.Module):
    """
    Class for handling the input embeddings of tokens.
    """

    def __init__(
            self, 
            model_dimension: int, 
            vocab_size: int
        ) -> None:
        """Initializing the InputEmbeddings object."""
        super().__init__()

        # Initialize parameters.
        self.model_dimension = model_dimension
        self.vocab_size = vocab_size

        # Tablica koja uči vektorsku reprezentaciju svakog tokena iz vokabulara
        self.embedding = nn.Embedding(vocab_size, model_dimension)

    def forward(self, x) -> torch.Tensor:
        """
        Translates the token into it's embedding.
        """
        return self.embedding(x) * math.sqrt(self.model_dimension)
    
    
class PositionalEncoding(nn.Module):
    """
    Class for handling the positional embeddings of tokens.
    """

    def __init__(
            self, 
            model_dimension: int, 
            context_size: int, 
            dropout: float
        ) -> None:
        """Initializing the PositionalEncoding object."""
        super().__init__()

        # Initialize parameters
        self.model_dimension = model_dimension
        self.context_size = context_size
        self.dropout = nn.Dropout(dropout)
        
        # Placeholder matrix for positional encodings
        positional_encodings = torch.zeros(context_size, model_dimension) # (context_size, model_dimension)
        # Vector [0, 1, 2, ..., context_size - 1] – indeks pozicije u rečenici
        position = torch.arange(0, context_size, dtype = torch.float).unsqueeze(1) # (context_size, 1)
        # Skalarni faktori iz formule u "Attention is all you need" radu
        div_term = torch.exp(torch.arange(0, model_dimension, 2).float() * (-math.log(10000.0) / model_dimension)) # (model_dimension / 2)
        # Sinus ide na parne koordinate vektora
        positional_encodings[:, 0::2] = torch.sin(position * div_term) # sin(position * 10000 ^ (2i / model_dimension))
        # Kosinus ide na neparne koordinate vektora
        positional_encodings[:, 1::2] = torch.cos(position * div_term) # cos(position * 10000 ^ (2i / model_dimension))

        # Dodajemo batch dimenziju: sada je oblik (1, context_size, model_dimension)
        positional_encodings = positional_encodings.unsqueeze(0) # (1, context_size, model_dimension)

        # register_buffer → čuva tensor u modelu, ali se ne trenira (nije parametar)
        self.register_buffer('pe', positional_encodings)

    def forward(self, x):
        """
        Adds the positional encodings to input embeddings of a 
        given token, and applies dropout for regularization.
        """
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
        
class LayerNormalization(nn.Module):
    """
    Class for handling the normalization of vectors in a given layer.
    """

    def __init__(
            self, 
            features: int, 
            eps: float = 10**-6
        ) -> None:
        """Initializing the LayerNormalization object."""
        super().__init__()

        # Initialize parameters.
        # Eps je mali broj za numeričku stabilnost (da ne delimo sa 0)
        self.eps = eps
        # Alpha i bias su trenirajući parametri koji pomeraju/skaliraju normalizovan izlaz
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        """
        Applies the normalization to a given embedding.
        """
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
    
class MultiHeadAttentionBlock(nn.Module):
    """
    Class for handling the multihead attention.
    """

    def __init__(
            self, 
            model_dimension: int, 
            heads: int, 
            dropout: float
        ) -> None:
        """Initializing the MultiHeadAttentionBlock object."""
        super().__init__()

        # Initialize parameters.
        self.model_dimension = model_dimension
        self.heads = heads
        self.dropout = nn.Dropout(dropout)

        # Provera: dimenzija modela mora da se deli na broj glava
        assert model_dimension % heads == 0, "model_dimension is not divisible by the number of heads."

        # Svaka glava vidi samo deo vektora (head_dimension)
        self.head_dimension = model_dimension // heads

        # Linearne projekcije za Q, K i V (uče se tokom treninga)
        self.w_q = nn.Linear(model_dimension, model_dimension)
        self.w_k = nn.Linear(model_dimension, model_dimension)
        self.w_v = nn.Linear(model_dimension, model_dimension)

        # Projekcija za spajanje svih glava nazad u isti prostor
        self.w_o = nn.Linear(model_dimension, model_dimension)

    @staticmethod
    def attention(
            query, 
            key, 
            value, 
            mask, 
            dropout: nn.Dropout
        ):
        """
        Perform a masked multi head attention on the given matrices.
        Apply the formula from the "Attention is all you need".
        Attention(Q, K, V) = softmax(QK^T / sqrt(head_dimension))V
        head_i = Attention(QWi^Q, KWi^K, VWi^V)
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
        """
        head_dimension = query.shape[-1]

        # Glavni deo: QK^T / sqrt(d_k) → matrica sličnosti između tokena
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dimension) # (batch, heads, context_size, context_size)

        # Maskiramo zabranjene pozicije (padding ili "budućnost" u decoder-u)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        # Softmax pretvara skorove u verovatnoće po redu (gde gledamo ka svim tokenima)
        attention_scores = attention_scores.softmax(dim = -1)

        # Dropout dodatno regularizuje attention matrice
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Vraćamo novi "kontekstualizovani" vektor + same attention skorove
        return (attention_scores @ value), attention_scores # (batch, heads, context_size, head_dimension)

    def forward(self, q, k, v, mask):
        """
        Apply the multi-headed attention to the given inputs.
        Can be used for both encoder and decoder, the inputs determine which.
        """

        # Linearne projekcije ulaza → Q', K', V'
        query = self.w_q(q) # (batch, context_size, model_dimension) --> (batch, context_size, model_dimension)
        key = self.w_k(k) # (batch, context_size, model_dimension) --> (batch, context_size, model_dimension)
        value = self.w_v(v) # (batch, context_size, model_dimension) --> (batch, context_size, model_dimension)

        # Preuređivanje tenzora tako da eksplicitno odvojimo glave (batch, heads, seq, head_dim)
        # (batch, context_size, model_dimension) --> (batch, context_size, heads, head_dimension) --> (batch, heads, context_size, head_dimension)
        query = query.view(query.shape[0], query.shape[1], self.heads, self.head_dimension).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.heads, self.head_dimension).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.heads, self.head_dimension).transpose(1, 2)

        # Izvršavamo attention za sve glave odjednom
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
         
        # Vraćamo nazad oblik (batch, seq_len, model_dimension) spajanjem svih glava
        # (batch, heads, context_size, head_dimension) --> (batch, context_size, heads, head_dimension) -> (batch, context_size, model_dimension)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.heads * self.head_dimension)

        # Završna linearna projekcija preko svih glava
        # (batch, context_size, model_dimension) --> (batch, context_size, model_dimension)
        return self.w_o(x)
    
    
class FeedForwardBlock(nn.Module):
    """
    Class for handling the feed forward neural networks.
    """

    def __init__(
            self, 
            model_dimension: int, 
            feed_forward_dimension: int, 
            dropout: float
        ) -> None:
        """Initializing the FeedForwardBlock object."""
        super().__init__()
        
        # Dva linearna sloja sa ReLU između: uvodi nelinearnost po tokenu
        self.linear_1 = nn.Linear(model_dimension, feed_forward_dimension)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(feed_forward_dimension, model_dimension)

    def forward(self, x):
        """
        Apply the feed forward to the given input.
        FNN(x) = ReLU(xW_1 + b_1)W_2 + b_2
        """
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
    
class ResidualConnection(nn.Module):
    """
    Class for handling the residual connections in the model.
    Serves as a connection between the input and the LayerNormalization object.
    """

    def __init__(
            self, 
            features: int, 
            dropout: float
        ) -> None:
        """Initializing the ResidualConnection object."""
        super().__init__()

        # Dropout + LayerNorm koje se ponavljaju kroz ceo model
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        """Apply the layer normalization to the input and input passed through the sublayer."""
        # Praksa: prvo normiramo ulaz, zatim primenimo "sublayer" i dodamo rezidualno x
        return x + self.dropout(sublayer(self.norm(x)))
    
    
class EncoderBlock(nn.Module):
    """
    Class for handling one iteration of the encoder.
    """

    def __init__(
            self, 
            features: int, 
            self_attention_block: MultiHeadAttentionBlock, 
            feed_forward_block: FeedForwardBlock, 
            dropout: float
        ) -> None:
        """Initializing the EncoderBlock object."""
        super().__init__()

        # Jedan encoder blok = self-attention + feed-forward, svaki sa svojom rezidualnom vezom
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # Two residual connections, one for feed forward, and one for self attention.
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, source_mask):
        """Generate the output of a single iteration of the encoder."""

        # Prvo self-attention nad ulazom (sa maskom koja skriva PAD tokene)
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, source_mask))

        # Zatim feed-forward nad svakim tokenom zasebno
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x
    
    
class Encoder(nn.Module):
    """
    Class for handling all the iterations of the encoder.
    """

    def __init__(
            self, 
            features: int, 
            layers: nn.ModuleList
        ) -> None:
        """Initializing the Encoder object."""
        super().__init__()

        # Lista svih EncoderBlock-ova (broj slojeva)
        self.layers = layers
        # Završna LayerNorm preko poslednjeg izlaza iz encoder-a
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        """Generate the output of the encoder."""

        # Sekvencijalno prolazimo kroz sve encoder blokove
        for layer in self.layers:
            x = layer(x, mask)

        # Normalize the output.
        return self.norm(x)
    
    
class DecoderBlock(nn.Module):
    """
    Class for handling one iteration of the decoder.
    """

    def __init__(
            self, 
            features: int, 
            self_attention_block: MultiHeadAttentionBlock, 
            cross_attention_block: MultiHeadAttentionBlock, 
            feed_forward_block: FeedForwardBlock, 
            dropout: float
        ) -> None:
        """Initializing the DecoderBlock object."""
        super().__init__()

        # Blok dekodera ima self-attention, cross-attention i feed-forward
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        # Tri rezidualne veze – po jedna za svaki od ova tri podsloja
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, source_mask, target_mask):
        """Generate the output of a single iteration of a decoder."""

        # Self-attention nad ciljnim jezikom (uz causal + padding masku)
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, target_mask))

        # Cross-attention: Q dolazi iz decodera, a K i V iz encoder izlaza
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, source_mask))

        # Na kraju feed-forward po tokenu
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x
    
    
class Decoder(nn.Module):
    """
    Class for handling all the iterations of the decoder.
    """

    def __init__(
            self, 
            features: int, 
            layers: nn.ModuleList
        )-> None:
        """Initializing the decoder object."""
        super().__init__()

        # Lista svih DecoderBlock-ova (broj slojeva)
        self.layers = layers
        # Završna normalizacija na kraju dekodera
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, source_mask, target_mask):
        """Generate the output of the decoder."""

        # Prolazimo kroz sve slojeve dekodera
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)

        # Normalize the output.
        return self.norm(x)
    
    
class ProjectionLayer(nn.Module):
    """
    Class for handling the output of the transformer and converting it to the appropriate token.
    """

    def __init__(
            self, 
            model_dimension: int, 
            vocab_size: int
        ) -> None:
        """Initializing the ProjectionLayer object."""
        super().__init__()

        # Linearni sloj: iz embedding prostora u dimenziju vokabulara
        self.proj = nn.Linear(model_dimension, vocab_size)

    def forward(self, x):
        """Convert the input to a vector of probabilities."""
        return torch.log_softmax(self.proj(x), dim = -1)
    
    
class Transformer(nn.Module):
    """
    Class for handling all of the transformer.
    """
    
    def __init__(
            self, 
            encoder: Encoder, 
            decoder: Decoder, 
            source_embed: InputEmbeddings, 
            target_embed: InputEmbeddings, 
            source_pos: PositionalEncoding, 
            target_pos: PositionalEncoding, 
            projection_layer: ProjectionLayer
        ) -> None:
        """Initializing the Transformer object."""
        super().__init__()

        # Ovde samo "sklapamo" sve prethodne blokove u jedan model
        self.encoder = encoder
        self.decoder = decoder
        self.source_embed = source_embed
        self.target_embed = target_embed
        self.source_pos = source_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer

    def encode(self, source, source_mask):
        """Generate the output of the encoder."""
        source = self.source_embed(source)
        source = self.source_pos(source)
        return self.encoder(source, source_mask)
    
    def decode(self, encoder_output, source_mask, target, target_mask):
        """Generate the output of the decoder."""
        target = self.target_embed(target)
        target = self.target_pos(target)
        return self.decoder(target, encoder_output, source_mask, target_mask)
    
    def project(self, x):
        """Generate the output of the transformer."""
        return self.projection_layer(x)
    
    
def build_transformer(
        source_vocab_size: int, 
        target_vocab_size: int, 
        source_context_size: int, 
        target_context_size: int, 
        model_dimension: int = 512, 
        number_of_blocks: int = 6, 
        heads: int = 8, 
        dropout: float = 0.1, 
        feed_forward_dimension: int = 2048
    ) -> Transformer:
    """Build the transformer with the provided parameters.

    Args:
        source_vocab_size (int): Size of the vocabulary in the source language.
        target_vocab_size (int): Size of the vocabulary in the target language.
        source_context_size (int): Maximum allowed sentence size in the source language.
        target_context_size (int): Maximum allowed sentence size in the target language.
        model_dimension (int, optional): Dimension of the embedding space and thus the model dimension. Defaults to 512.
        number_of_blocks (int, optional): Number of encoder and decoder blocks in the transformer. Defaults to 6.
        heads (int, optional): Number of heads for multihead attention. Defaults to 8.
        dropout (float, optional): Rate of dropout. Dropout removes certain weights for regularization sake. Defaults to 0.1.
        feed_forward_dimension (int, optional): Dimension of the hidden layer in feed forward network. Defaults to 2048.

    Returns:
        Transformer: An initialized transformer with the specified parameters.
    """
    # Embedding slojevi za izvorni i ciljni jezik
    # Source embeddings act as input to encoder, target embeddings act as input to decoder.
    source_embed = InputEmbeddings(model_dimension, source_vocab_size)
    target_embed = InputEmbeddings(model_dimension, target_vocab_size)

    # Positional encoding slojevi – zavise samo od context_size i dimenzije modela
    source_pos = PositionalEncoding(model_dimension, source_context_size, dropout)
    target_pos = PositionalEncoding(model_dimension, target_context_size, dropout)

    # Pravimo listu encoder blokova (svaki ima svoj attention i feed-forward)
    encoder_blocks = []
    for _ in range(number_of_blocks):
        encoder_self_attention_block = MultiHeadAttentionBlock(model_dimension, heads, dropout)
        feed_forward_block = FeedForwardBlock(model_dimension, feed_forward_dimension, dropout)
        encoder_block = EncoderBlock(model_dimension, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Pravimo listu decoder blokova – svaki ima self-attention + cross-attention + feed-forward
    decoder_blocks = []
    for _ in range(number_of_blocks):
        decoder_self_attention_block = MultiHeadAttentionBlock(model_dimension, heads, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(model_dimension, heads, dropout)
        feed_forward_block = FeedForwardBlock(model_dimension, feed_forward_dimension, dropout)
        decoder_block = DecoderBlock(model_dimension, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Sastavljamo Encoder i Decoder objekte iz lista blokova
    encoder = Encoder(model_dimension, nn.ModuleList(encoder_blocks))
    decoder = Decoder(model_dimension, nn.ModuleList(decoder_blocks))

    # Projekcioni sloj prevodi embedding vektore nazad u raspodelu po vokabularu
    projection_layer = ProjectionLayer(model_dimension, target_vocab_size)

    # Konačni Transformer objekat – koristi se u treningu i inferenci
    transformer = Transformer(encoder, decoder, source_embed, target_embed, source_pos, target_pos, projection_layer)

    # Xavier inicijalizacija težina za sve slojeve dimenzije > 1
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


def get_model(
        config, 
        source_vocab_size: int, 
        target_vocab_size: int
    ) -> Transformer:
    """
    Build the transformer from the config file with given vocabulary sizes,
    using the model.py::build_transformer function.

    A pretty unnecessary function in my opinion.

    Args:
        config: A config file.
        source_vocab_size (int): Vocabulary size of the source language (language of the encoder).
        target_vocab_size (int): Vocabulary size of the target language (language of the decoder).

    Returns:
        Transformer: An initialized transformer model.
    """
    model = build_transformer(
        source_vocab_size = source_vocab_size, 
        target_vocab_size = target_vocab_size, 
        source_context_size = config['context_size'], 
        target_context_size = config['context_size'], 
        model_dimension = config['model_dimension']
        )
    
    return model
    