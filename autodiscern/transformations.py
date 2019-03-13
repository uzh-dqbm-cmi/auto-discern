import multiprocessing as mp
from bs4 import BeautifulSoup
from typing import Callable, Dict, List


TransformType = Callable[[str], str]


class Transformer:
    """Run a set of transforms on any input. """

    def __init__(self, transforms: List[TransformType], num_cores=8):
        self.transforms = transforms
        self.num_cores = num_cores

    def _apply_transforms_to_str(self, content: str) -> str:
        """Apply list all transforms to str. """

        for f in self.transforms:
            content = f(content)
        return content

    def _transform_worker(self, obj: Dict) -> Dict:
        """Create a new transformed object. """
        transformed_obj = {key: obj[key] for key in obj if key != 'content'}
        transformed_obj['content'] = self._apply_transforms_to_str(obj['content'])
        return transformed_obj

    def apply(self, input_list: List[Dict]) -> List:
        """Run all transforms on input_list in parallel. """
        pool = mp.Pool(self.num_cores)
        results = pool.map(self._transform_worker, (i for i in input_list))
        pool.close()
        pool.join()
        return results


def remove_html(x: str, replacement_char='. ') -> str:
    """Replace all html tags with replacement_char. """
    return BeautifulSoup(x, features="html.parser").get_text(separator=replacement_char)


def remove_selected_html(x: str, replacement_char='. ') -> str:
    print("I don't know how to do this!")
    return x


def replace_problem_chars(x: str, replacement_char=' ') -> str:
    """Replace all problem chars with replacement_char. """
    problem_chars = [
        "\n",
        "\t",
    ]
    for p in problem_chars:
        x = x.replace(p, replacement_char)
    return x
