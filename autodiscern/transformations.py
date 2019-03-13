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


def remove_selected_html(x: str) -> str:
    """Remove all tags except for tags_to_keep, and replace the contents of link tags with LINK"""

    soup = BeautifulSoup(x, features="html.parser")
    tags_to_keep_attr = ['a']
    tags_to_keep = {'a', 'h1', 'h2', 'h3', 'h4'}
    tags_to_remove = set([tag.name for tag in soup.find_all()]) - tags_to_keep

    for tag in soup.find_all(True):
        if tag.name not in tags_to_keep_attr:
            # clear all attributes
            tag.attrs = {}
        else:
            attrs = dict(tag.attrs)
            for attr in attrs:
                if attr not in ['src', 'href']:
                    del tag.attrs[attr]
                else:
                    tag.attrs[attr] = 'LINK'

    text = str(soup)

    # all tags to remove have been cleared down to their bare tag form without attributes, and can be found/replaced
    for t in tags_to_remove:
        text = text.replace('<{}>'.format(t), '').replace('</{}>'.format(t), '').replace('<{}/>'.format(t), '')

    return text


def replace_problem_chars(x: str, replacement_char=' ') -> str:
    """Replace all problem chars with replacement_char. """
    problem_chars = [
        "\n",
        "\t",
    ]
    for p in problem_chars:
        x = x.replace(p, replacement_char)
    return x
