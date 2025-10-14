import numpy as np
from sklearn.preprocessing import normalize


def generate_test_scenarios():
    """Generate different clustering scenarios with semantic word groups to test SCCI behavior."""

    scenarios = {}

    # Define semantic word groups
    word_groups = {
        'fruits': ['apple', 'banana', 'orange', 'grape', 'strawberry', 'pineapple', 'mango', 'peach',
                   'pear', 'cherry', 'watermelon', 'kiwi', 'blueberry', 'raspberry', 'apricot',
                   'plum', 'lemon', 'lime', 'coconut', 'avocado'],
        'animals': ['dog', 'cat', 'elephant', 'lion', 'tiger', 'giraffe', 'zebra', 'monkey',
                    'bear', 'wolf', 'fox', 'rabbit', 'deer', 'horse', 'cow', 'sheep',
                    'pig', 'goat', 'chicken', 'duck'],
        'colors': ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown',
                   'black', 'white', 'gray', 'violet', 'indigo', 'turquoise', 'magenta',
                   'cyan', 'maroon', 'navy', 'olive', 'silver'],
        'vehicles': ['car', 'truck', 'bicycle', 'motorcycle', 'bus', 'train', 'airplane', 'boat',
                     'ship', 'helicopter', 'scooter', 'van', 'taxi', 'subway', 'tram',
                     'ferry', 'rocket', 'skateboard', 'ambulance', 'firetruck'],
        'professions': ['doctor', 'teacher', 'engineer', 'lawyer', 'nurse', 'chef', 'artist', 'musician',
                        'writer', 'programmer', 'pilot', 'firefighter', 'police', 'architect', 'dentist',
                        'scientist', 'journalist', 'photographer', 'designer', 'accountant'],
        'sports': ['football', 'basketball', 'tennis', 'soccer', 'baseball', 'hockey', 'golf', 'swimming',
                   'running', 'cycling', 'boxing', 'wrestling', 'skiing', 'surfing', 'volleyball',
                   'badminton', 'cricket', 'rugby', 'bowling', 'archery']
    }

    # Scenario 1: Perfect clustering - tight and well-separated
    print("Generating Scenario 1: Perfect clustering")
    np.random.seed(42)

    # Create tight, well-separated clusters for fruits, animals, colors
    cluster_fruits = normalize(np.random.randn(20, 50) * 0.1 + np.array([10] + [0] * 49).reshape(1, -1))
    cluster_animals = normalize(np.random.randn(20, 50) * 0.1 + np.array([0, 10] + [0] * 48).reshape(1, -1))
    cluster_colors = normalize(np.random.randn(20, 50) * 0.1 + np.array([0, 0, 10] + [0] * 47).reshape(1, -1))

    scenarios['perfect'] = {
        'embeddings': np.vstack([cluster_fruits, cluster_animals, cluster_colors]),
        'labels': (['fruits'] * 20 + ['animals'] * 20 + ['colors'] * 20),
        'words': (word_groups['fruits'][:20] + word_groups['animals'][:20] + word_groups['colors'][:20]),
        'description': 'Perfect: Tight clusters, well-separated (fruits, animals, colors)'
    }

    # Scenario 2: Good clustering - tight but some overlap
    print("Generating Scenario 2: Good clustering")
    np.random.seed(43)
    cluster_vehicles = normalize(np.random.randn(20, 50) * 0.2 + np.array([5, 1] + [0] * 48).reshape(1, -1))
    cluster_professions = normalize(np.random.randn(20, 50) * 0.2 + np.array([1, 5] + [0] * 48).reshape(1, -1))
    cluster_sports = normalize(np.random.randn(20, 50) * 0.2 + np.array([0, 0, 5] + [0] * 47).reshape(1, -1))

    scenarios['good'] = {
        'embeddings': np.vstack([cluster_vehicles, cluster_professions, cluster_sports]),
        'labels': (['vehicles'] * 20 + ['professions'] * 20 + ['sports'] * 20),
        'words': (word_groups['vehicles'][:20] + word_groups['professions'][:20] + word_groups['sports'][:20]),
        'description': 'Good: Tight clusters, moderate separation (vehicles, professions, sports)'
    }

    # Scenario 3: Poor clustering - loose clusters
    print("Generating Scenario 3: Poor clustering - loose")
    np.random.seed(44)
    cluster_fruits_loose = normalize(np.random.randn(20, 50) * 1.0 + np.array([2, 0] + [0] * 48).reshape(1, -1))
    cluster_animals_loose = normalize(np.random.randn(20, 50) * 1.0 + np.array([0, 2] + [0] * 48).reshape(1, -1))
    cluster_colors_loose = normalize(np.random.randn(20, 50) * 1.0 + np.array([1, 1] + [0] * 48).reshape(1, -1))

    scenarios['loose'] = {
        'embeddings': np.vstack([cluster_fruits_loose, cluster_animals_loose, cluster_colors_loose]),
        'labels': (['fruits'] * 20 + ['animals'] * 20 + ['colors'] * 20),
        'words': (word_groups['fruits'][:20] + word_groups['animals'][:20] + word_groups['colors'][:20]),
        'description': 'Poor: Loose clusters, some separation (fruits, animals, colors)'
    }

    # Scenario 4: Poor clustering - high overlap
    print("Generating Scenario 4: Poor clustering - overlapping")
    np.random.seed(45)
    base_direction = np.array([1, 1, 1] + [0] * 47).reshape(1, -1)
    cluster_vehicles_overlap = normalize(np.random.randn(20, 50) * 0.3 + base_direction)
    cluster_professions_overlap = normalize(np.random.randn(20, 50) * 0.3 + base_direction * 1.1)
    cluster_sports_overlap = normalize(np.random.randn(20, 50) * 0.3 + base_direction * 0.9)

    scenarios['overlap'] = {
        'embeddings': np.vstack([cluster_vehicles_overlap, cluster_professions_overlap, cluster_sports_overlap]),
        'labels': (['vehicles'] * 20 + ['professions'] * 20 + ['sports'] * 20),
        'words': (word_groups['vehicles'][:20] + word_groups['professions'][:20] + word_groups['sports'][:20]),
        'description': 'Poor: Tight clusters but highly overlapping (vehicles, professions, sports)'
    }

    # Scenario 5: Worst case - random assignment (mixed categories)
    print("Generating Scenario 5: Random clustering")
    np.random.seed(46)
    random_embeddings = normalize(np.random.randn(60, 50))

    # Mix words from different categories randomly
    all_words = (word_groups['fruits'][:10] + word_groups['animals'][:10] +
                 word_groups['colors'][:10] + word_groups['vehicles'][:10] +
                 word_groups['professions'][:10] + word_groups['sports'][:10])

    # Create random labels that don't match semantic groups
    np.random.seed(46)
    random_labels_idx = np.random.randint(0, 3, 60)
    random_labels = ['cluster_' + str(i) for i in random_labels_idx]

    scenarios['random'] = {
        'embeddings': random_embeddings,
        'labels': random_labels,
        'words': all_words,
        'description': 'Worst: Random cluster assignment (mixed semantic categories)'
    }

    # Scenario 6: Mixed quality - some good, some bad clusters
    print("Generating Scenario 6: Mixed quality")
    np.random.seed(47)
    # One tight cluster (fruits)
    cluster_fruits_tight = normalize(np.random.randn(20, 50) * 0.1 + np.array([10] + [0] * 49).reshape(1, -1))
    # One loose cluster (animals)
    cluster_animals_mixed = normalize(np.random.randn(20, 50) * 1.0)
    # One overlapping with first (colors, but close to fruits)
    cluster_colors_overlap = normalize(np.random.randn(20, 50) * 0.2 + np.array([9] + [0] * 49).reshape(1, -1))

    scenarios['mixed'] = {
        'embeddings': np.vstack([cluster_fruits_tight, cluster_animals_mixed, cluster_colors_overlap]),
        'labels': (['fruits'] * 20 + ['animals'] * 20 + ['colors'] * 20),
        'words': (word_groups['fruits'][:20] + word_groups['animals'][:20] + word_groups['colors'][:20]),
        'description': 'Mixed: Fruits (tight), animals (loose), colors (overlapping with fruits)'
    }

    # Scenario 7: Many singleton clusters (individual words)
    # print("Generating Scenario 7: Many singletons")
    # np.random.seed(48)
    # # Few normal clusters (fruits and animals)
    # cluster_fruits_small = normalize(np.random.randn(5, 50) * 0.1 + np.array([5] + [0] * 49).reshape(1, -1))
    # cluster_animals_small = normalize(np.random.randn(5, 50) * 0.1 + np.array([0, 5] + [0] * 48).reshape(1, -1))
    # # Many singletons (individual words from different categories)
    # singletons = normalize(np.random.randn(20, 50))
    #
    # # Create individual word labels for singletons
    # singleton_words = []
    # singleton_labels = []
    # for i, category in enumerate(['colors', 'vehicles', 'professions', 'sports']):
    #     for j in range(5):
    #         singleton_words.append(word_groups[category][j])
    #         singleton_labels.append(f'singleton_{category}_{j}')
    #
    # scenarios['singletons'] = {
    #     'embeddings': np.vstack([cluster_fruits_small, cluster_animals_small, singletons]),
    #     'labels': (['fruits'] * 5 + ['animals'] * 5 + singleton_labels),
    #     'words': (word_groups['fruits'][:5] + word_groups['animals'][:5] + singleton_words),
    #     'description': 'Edge case: Few clusters (fruits, animals) + many singleton words'
    # }

    # Scenario 8: Hierarchical structure (sub-categories)
    print("Generating Scenario 8: Hierarchical clustering")
    np.random.seed(49)

    # Create sub-clusters within animals: mammals vs birds
    mammals = ['dog', 'cat', 'elephant', 'lion', 'tiger', 'bear', 'wolf', 'fox', 'rabbit', 'deer']
    birds = ['chicken', 'duck', 'eagle', 'sparrow', 'penguin', 'owl', 'parrot', 'flamingo', 'peacock', 'robin']

    # Mammals cluster
    cluster_mammals = normalize(np.random.randn(10, 50) * 0.15 + np.array([8, 2] + [0] * 48).reshape(1, -1))
    # Birds cluster (close to mammals but distinguishable)
    cluster_birds = normalize(np.random.randn(10, 50) * 0.15 + np.array([7, 3] + [0] * 48).reshape(1, -1))
    # Vehicles cluster (well separated)
    cluster_vehicles_hier = normalize(np.random.randn(20, 50) * 0.1 + np.array([0, 0, 8] + [0] * 47).reshape(1, -1))

    scenarios['hierarchical'] = {
        'embeddings': np.vstack([cluster_mammals, cluster_birds, cluster_vehicles_hier]),
        'labels': (['mammals'] * 10 + ['birds'] * 10 + ['vehicles'] * 20),
        'words': (mammals + birds + word_groups['vehicles'][:20]),
        'description': 'Hierarchical: Animals (mammals vs birds sub-clusters) + vehicles'
    }

    return scenarios


def print_scenario_summary(scenarios):
    """Print a summary of all generated scenarios."""
    print("\n" + "=" * 80)
    print("CLUSTERING SCENARIOS SUMMARY")
    print("=" * 80)

    for name, scenario in scenarios.items():
        print(f"\n{name.upper()}:")
        print(f"  Description: {scenario['description']}")
        print(f"  Total samples: {len(scenario['embeddings'])}")
        print(f"  Unique labels: {len(set(scenario['labels']))}")
        print(f"  Sample words: {scenario['words'][:5]}...")
        print(f"  Label distribution: {dict(zip(*np.unique(scenario['labels'], return_counts=True)))}")


# Example usage
if __name__ == "__main__":
    scenarios = generate_test_scenarios()
    print_scenario_summary(scenarios)

    # Access individual scenarios
    perfect_scenario = scenarios['perfect']
    print(f"\nPerfect scenario embeddings shape: {perfect_scenario['embeddings'].shape}")
    print(f"Perfect scenario sample words: {perfect_scenario['words'][:10]}")
    print(f"Perfect scenario sample labels: {perfect_scenario['labels'][:10]}")