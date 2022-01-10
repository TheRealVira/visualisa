#!/usr/bin/env python3
from PIL import ImageChops, ImageStat, Image, ImageDraw
import argparse, random, pathlib

# Constants
GENERATIONS_DIR = "./generations/"

# Arguments
parser = argparse.ArgumentParser(
    "visualisa",
    description="A machine learning algorithm which will take your input image and try to recrete it using circles.",
    epilog="And that's how you'd use visualisa.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--input", help="Your input bmp - smaller is better.", type=str)
parser.add_argument(
    "--iterations",
    help="Iteration limit - this will determin the iteration how many times new generations get populationulated.",
    type=int,
    default=100000,
)
parser.add_argument(
    "--population",
    help="population limit - sets how large one new population will be.",
    type=int,
    default=25,
)
parser.add_argument("--children", help="Children per population.", type=int, default=5)
parser.add_argument(
    "--chromosomes", help="Number of chromosomes (shapes).", type=int, default=125
)
parser.add_argument("--mutation", help="Mutation rate.", type=float, default=0.05)
parser.add_argument(
    "--backgroundcolor", help="Backgroundcolor.", type=str, default="black"
)
parser.add_argument("--seed", help="Starting seed for algorithm.", type=int)
parser.add_argument(
    "--minvert", help="Minimum vertices for a chromosome.", type=int, default=3
)
parser.add_argument(
    "--maxvert", help="Maximum vertices for a chromosome.", type=int, default=4
)
args = parser.parse_args()

# Create structure
pathlib.Path(GENERATIONS_DIR).mkdir(parents=True, exist_ok=True)

# Populating variables
GOAL_IMG = Image.open(args.input).convert('RGB')
GOAL_SUFFIX = pathlib.Path(args.input).suffix
MAX_SEQUENCE_SECTION_SIZE = int(args.maxvert) * 5
size_x, size_y = GOAL_IMG.size
CHOICE_WEIGHTS = [i * 0.1 for i in range(10, 0, -1)]

if args.seed is not None:
    random.seed(int(args.seed))


class chromosome(object):
    def __init__(self, inheritGene=None):
        if inheritGene is None or float(args.mutation) > random.random():
            self.color = self.generateColor()
            self.vertices = [
                self.generateVertex()
                for i in range(random.randint(int(args.minvert), int(args.maxvert)))
            ]
        else:
            self.__dict__.update(inheritGene)

        self.proteins = list(vars(self).items())

    def generateVertex(self):
        return (random.randrange(size_x), random.randrange(size_y))

    def generateColor(self):
        return [
            random.randrange(256),
            random.randrange(256),
            random.randrange(256),
            random.randrange(68),
        ]


class entity(object):
    def __init__(self, predator0=None, predator1=None):
        self.chromosomes = []
        if predator0 and predator1:
            self.generate(self.breed(predator0, predator1))
        else:
            self.chromosomes += [chromosome() for i in range(int(args.chromosomes))]

        self.image = self.render()
        self.fitness = self.getFitness()

    def render(self):
        image = Image.new(GOAL_IMG.mode, GOAL_IMG.size, args.backgroundcolor)
        drawer = ImageDraw.Draw(image, "RGBA")
        for aChromosome in self.chromosomes:
            drawer.polygon(
                aChromosome.vertices,
                fill=tuple(aChromosome.color),
            )
        del drawer
        return image.convert("RGB")

    def getDNA(self):
        return [
            protein
            for aChromosome in self.chromosomes
            for protein in aChromosome.proteins
        ]

    def breed(self, p0, p1):
        dna0 = p0.getDNA()
        combinedDNA = p1.getDNA()[:]
        for i in random.sample(
            list(range(len(dna0))), round(random.random() * len(dna0))
        ):
            combinedDNA[i] = dna0[i]
        return zip(*[iter(combinedDNA)] * len(p0.chromosomes[0].proteins))

    def saveImage(self, filename):
        self.image.save(filename, GOAL_IMG.format)

    def getFitness(self):
        stat = ImageStat.Stat(ImageChops.difference(self.image, GOAL_IMG))
        return sum(stat.mean) / (len(stat.mean) * 255) * 100

    def generate(self, children):
        self.chromosomes = [chromosome(dict(child)) for child in children]


def initialize(population):
    (
        population := population + [entity() for i in range(int(args.population) * 2)]
    ).sort(key=lambda x: x.fitness)
    return population[: int(args.population)]


def evolve(population):
    weightedP = random.choices(
        population[: len(CHOICE_WEIGHTS)], CHOICE_WEIGHTS, k=len(CHOICE_WEIGHTS) * 2
    )
    for i in range(int(args.children)):
        population.append(entity(weightedP[i], weightedP[i + len(CHOICE_WEIGHTS) - 1]))

    population.sort(key=lambda x: x.fitness)
    return population[: int(args.population)]


if __name__ == "__main__":
    population = initialize([])
    for i in range(0, int(args.iterations)):
        population = evolve(population)
        if i % 1000 == 0:
            population[0].saveImage(GENERATIONS_DIR + str(i) + GOAL_SUFFIX)
        if i % 10 == 0:
            print(
                "Average fitness:"
                + str(
                    100
                    - sum(map(lambda x: x.fitness, population)) / int(args.population)
                )
                + "\t/ 100%"
            )

    population[0].saveImage(GENERATIONS_DIR + "best" + GOAL_SUFFIX)
