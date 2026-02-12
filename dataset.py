import torch

from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset as HFDataset
from tokenizers import Tokenizer
from translate.storage.tmx import tmxfile

from typing import Any, Dict


# Torch Dataset koji priprema parove rečenica (izvor → cilj) za transformer
class BilingualDataset(TorchDataset):
    """
    Wrapper class of Torch Dataset.
    Has to have methods __init__, __len__ and __getitem__ to function properly.
    """

    def __init__(
            self, 
            dataset: HFDataset, 
            source_tokenizer: Tokenizer, 
            target_tokenizer: Tokenizer, 
            source_language: str, 
            target_language: str, 
            context_size: int
        ) -> None:
        """Initializing the BilingualDataset object.

        Args:
            dataset (HFDataset): 
                HuggingFace dataset with columns id and translations.
                    id is the number of the current row.
                    translations is a dictionary with entries 'language': sentence,
                    which has at least two different languages.
            source_tokenizer (Tokenizer): Tokenizer for the source language.
            target_tokenizer (Tokenizer): Tokenizer for the target language.
            source_language (str): Source language for the translations.
            target_language (str): Target language for the translations.
            context_size (int): Maximum allowed length of a sentence (in either language).
        """
        super().__init__()

        # Initializing context size.
        self.context_size = context_size

        # Initializing the dataset.
        self.dataset = dataset

        # Initializing the tokenizers.
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

        # Initializing the languages.
        self.source_language = source_language
        self.target_language = target_language

        # Initializing the start of sentence, end of sentence and padding tokens.
        # Ovi specijalni tokeni su neophodni za rad sekvencijalnih modela
        # Start of sentence token signifies the beginning of a sentence.
        
        # Iz tokenizer-a uzima ID broja koji odgovara specijalnom tokenu [SOS] (isto za [EOS] i [PAD]).
        # Taj ID broja se pretvara u PyTorch tenzor (vektor dužine 1) tipa int64, da bi mogao direktno da se lepi (torch.cat) na ostale tenzore rečenice.
        # Piše se u uglastim zagradama ([...]) da bi self.sos_token imao oblik npr. [101], pa kasnije može da se konkatenira sa nizom tokena rečenice (koji je takođe 1D tenzor).
        self.sos_token = torch.tensor([source_tokenizer.token_to_id('[SOS]')], dtype = torch.int64)
        
        # End of sentence token signifies the end of a sentence.
        self.eos_token = torch.tensor([source_tokenizer.token_to_id('[EOS]')], dtype = torch.int64)

        # Padding token signifies the placeholder token for sentences shorter than context size, which fills the empty spaces.
        self.pad_token = torch.tensor([source_tokenizer.token_to_id('[PAD]')], dtype = torch.int64)


    def __len__(self) -> int:
        """
        Returns:
            int: Number of sentences in the dataset.
        """
        return len(self.dataset)
    
    
    def __getitem__(
            self, 
            index: int
        ) -> Dict[str, Any]:
        """Gets the row from the dictionary at a specified index.

        Args:
            index (int): Index at which to return the element from the list.

        Raises:
            ValueError: _description_

        Returns:
            Dict[str, Any]: A dictionary with 7 fields:
                encoder_input: 
                    Input to be fed to the encoder. 
                    Tensor of dimension (context_size)
                decoder_input:
                    Input to be fed to the decoder. 
                    Tensor of dimension (context_size)
                encoder_mask:
                    Mask for the encoder, that will mask any padding tokens.
                    Tensor of dimension (1, 1, context_size)
                decoder_mask:
                    Mask for the decoder, that will mask any padding tokens and won't allow predictions in the past.
                    Tensor of dimension (1, context_size, context_size)
                label:
                    Expected model output.
                    Tensor of dimension (context_size)
                source_text:
                    Sentence in the source language.
                target_text:
                    Sentence in the target language.
        """
        # Get the index-th row of the dataset.
        source_target_pair = self.dataset[index]

        # Iz baze čitamo originalnu i prevedenu rečenicu (stringove)
        source_text = source_target_pair['translation'][self.source_language]
        target_text = source_target_pair['translation'][self.target_language]

        # Prebacujemo rečenice u liste ID-jeva tokena pomoću tokenizer-a
        encoder_input_tokens = self.source_tokenizer.encode(source_text).ids
        decoder_input_tokens = self.target_tokenizer.encode(target_text).ids

        # Računamo koliko [PAD] tokena treba dodati da bismo došli do context_size
        # Encoder already has len(encoder_input_tokens), SOS and EOS.
        encoder_num_padding_tokens = self.context_size - len(encoder_input_tokens) - 2
        # Decoder already has len(decoder_input_tokens), and:
        #       SOS token for the input;
        #       EOS token for the label.
        decoder_num_padding_tokens = self.context_size - len(decoder_input_tokens) - 1
        
        # Ako rečenica ne staje u context_size, radije bacamo grešku nego da je sečemo
        if encoder_num_padding_tokens < 0 or decoder_num_padding_tokens < 0:
            raise ValueError("Sentence is too long!")
        
        # Encoder input je: [SOS] tokeni ulazne rečenice [EOS] [PAD] ... [PAD]
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * encoder_num_padding_tokens, dtype = torch.int64)
            ],
            dim = 0
        )

        # Decoder input je: [SOS] tokeni ciljne rečenice [PAD] ... [PAD]
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_input_tokens, dtype = torch.int64),
                torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype = torch.int64)
            ],
            dim = 0
        )

        # Label je pomerena verzija decoder ulaza:
        # tokeni ciljne rečenice [EOS] [PAD] ... [PAD]
        label = torch.cat(
            [
                torch.tensor(decoder_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype = torch.int64)
            ],
            dim = 0     # spaja po redovima
        )

        # Dodatna sigurnosna provera – sve sekvence MORAJU imati istu dužinu
        assert encoder_input.size(0) == self.context_size
        assert decoder_input.size(0) == self.context_size
        assert label.size(0) == self.context_size

        # Vraćamo sve što je potrebno za jedan trening korak transformera
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,
            "source_text" : source_text,
            "target_text" : target_text
        }

    
def causal_mask(size: int) -> torch.Tensor:
    """
    Generates a causal mask for the decoder. This is a triangular matrix that
    has all ones as inputs which deals with decoder having access to words that
    have not yet been translated.

    Args:
        size (int): Size of the mask matrix.

    Returns:
        torch.Tensor: Triangular matrix with all ones, of dimension (1, size, size).
    """
    # Gornja trougaona matrica iznad dijagonale (1 iznad glavne dijagonale)
    mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int)
    return mask == 0


def load_data(
        source_language: str, 
        target_language: str
    ) -> HFDataset:
    """
    Translates the .tmx file into a HFDataset with given languages.

    Args:
        source_language (str): Original language of the dataset.
        target_language (str): Translated language of the dateset.

    Returns:
        HFDataset: HuggingFace dataset with columns id and translations.
                    id - the number of the current row.
                    translations - a dictionary with entries 'language': sentence,
                    which has at least two different languages.
    """
    # Otvaramo .tmx fajl sa paralelnim rečenicama (izvorni jezik → ciljni jezik)
    with open(f"{source_language}-{target_language}.tmx", "rb") as fin:
        tmx_file = tmxfile(fin, "en", "sr_Cyrl")

    # Pripremamo strukturu podataka u formatu koji očekuje HuggingFace Dataset
    data = {'id' : [], 'translation': []}
    i = 0

    # Čitamo sve parove rečenica iz .tmx fajla i ubacujemo ih u dict
    for item in tmx_file.unit_iter():

        data["id"].append(str(i))
        i = i + 1

        data["translation"].append({f"{source_language}": item.source.strip('"\n').lower(), f"{target_language}": item.target.strip('"\n').lower()})

    # Na kraju pravimo HFDataset objekat iz običnog Python rečnika
    dataset = HFDataset.from_dict(data)

    return dataset
