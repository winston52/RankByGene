from torchvision import transforms


class TrainTransform(object):
    """Training-time transform for the teacher-student encoder. Returns a pair of
    views: a strongly augmented student view and a weakly augmented teacher view."""

    def __init__(self, input_image_size=224):
        # strong augmentation (student)
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])

        # weak augmentation (teacher)
        flip_and_rotate = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation((90, 90)),
        ])

        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.student_transform = transforms.Compose([
            transforms.Resize((input_image_size, input_image_size)),
            flip_and_color_jitter,
            normalize,
        ])
        self.teacher_transform = transforms.Compose([
            transforms.Resize((input_image_size, input_image_size)),
            flip_and_rotate,
            normalize,
        ])

    def __call__(self, image):
        return [self.student_transform(image), self.teacher_transform(image)]


class TestTransform(object):
    """Test/inference transform. Applies the same deterministic transform twice so
    the output matches the teacher-student encoder's two-view input format."""

    def __init__(self, input_image_size=224):
        self.transform = transforms.Compose([
            transforms.Resize((input_image_size, input_image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __call__(self, image):
        return [self.transform(image), self.transform(image)]
