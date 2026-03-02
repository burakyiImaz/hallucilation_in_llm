"""
Turkish Datasets Module - Test datasets and sample data for Turkish NLP
"""

from typing import List, Dict


class TurkishDatasets:
    """Turkish benchmark datasets and test samples"""
    
    @staticmethod
    def sample_qa_dataset() -> List[Dict]:
        """
        Sample Turkish Q&A dataset for quick testing
        
        Returns:
            List of Turkish Q&A samples
        """
        return [
            {
                'id': 'tr_1',
                'question': 'Türkiye\'nin başkenti neresidir?',
                'answer': 'Ankara',
                'category': 'coğrafya',
                'source': 'sample_tr'
            },
            {
                'id': 'tr_2',
                'question': 'Osmanlı İmparatorluğu ne zaman kuruldu?',
                'answer': '1299',
                'category': 'tarih',
                'source': 'sample_tr'
            },
            {
                'id': 'tr_3',
                'question': 'Atatürk kimdir?',
                'answer': 'Türkiye\'nin kurucusu ve ilk cumhubaşkanı',
                'category': 'tarih',
                'source': 'sample_tr'
            },
            {
                'id': 'tr_4',
                'question': 'Boğaz Köprüsü hangi iki kıtayı birleştirir?',
                'answer': 'Asya ve Avrupa',
                'category': 'coğrafya',
                'source': 'sample_tr'
            },
            {
                'id': 'tr_5',
                'question': 'Türkçe kaç tane ünlü harf vardır?',
                'answer': '8',
                'category': 'dil',
                'source': 'sample_tr'
            }
        ]
    
    @staticmethod
    def sample_hallucination_test_set() -> List[Dict]:
        """
        Turkish hallucination detection test set
        
        Returns:
            List of samples with ground truth for hallucination detection
        """
        return [
            {
                'question': 'Paris\'in başkenti nedir?',
                'model_answer': 'Paris Fransa\'nın başkentidir',
                'ground_truth': True,
                'expected_hallucination': False
            },
            {
                'question': 'Ay\'ın kaç ay vardır?',
                'model_answer': 'Ay\'ın 12 ayı vardır',
                'ground_truth': False,
                'expected_hallucination': True
            },
            {
                'question': 'Einstein kimdir?',
                'model_answer': 'Albert Einstein fizikçi ve matematikçidir',
                'ground_truth': True,
                'expected_hallucination': False
            },
            {
                'question': 'Dünya\'nın çevresi nedir?',
                'model_answer': 'Dünya\'nın çevresi 40.075 km\'dir',
                'ground_truth': True,
                'expected_hallucination': False
            }
        ]
    
    @staticmethod
    def sample_math_dataset() -> List[Dict]:
        """
        Turkish math problem dataset (Turkish GSM8K)
        
        Returns:
            List of Turkish math problems
        """
        return [
            {
                'id': 'tr_math_1',
                'problem': 'Ali 5 elma aldı, sonra 3 elma daha aldı. Ali\'nin toplam kaç elmasi var?',
                'answer': '8',
                'difficulty': 'easy'
            },
            {
                'id': 'tr_math_2',
                'problem': 'Bir kitap 25 lira, başka bir kitap ise 35 lira tutuyor. Iki kitabı alırsak toplam ne kadar para gerekir?',
                'answer': '60',
                'difficulty': 'easy'
            },
            {
                'id': 'tr_math_3',
                'problem': 'Bir sınıfta 30 öğrenci var. %20\'si kız ise, kaç kız öğrenci vardır?',
                'answer': '6',
                'difficulty': 'medium'
            }
        ]
    
    @staticmethod
    def dataset_info() -> Dict:
        """Get information about available Turkish datasets"""
        return {
            'available_datasets': [
                'sample_qa_dataset',
                'sample_hallucination_test_set',
                'sample_math_dataset'
            ],
            'supported_sources': [
                'TurQA',
                'TRC-QA',
                'Turkish GSM8K',
                'Turkish MMLU',
                'XQuAD-TR'
            ],
            'total_samples': len(TurkishDatasets.sample_qa_dataset()) + 
                           len(TurkishDatasets.sample_hallucination_test_set()) +
                           len(TurkishDatasets.sample_math_dataset())
        }
