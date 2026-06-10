def get_default_config(dataset):
    if dataset == 'HandWritten':
        return dict(
            lr=1e-4,
            aligned_ratio=0.5,
            beta=0.5,

            lambda_cons=1,
            k=5,

            seed=1  ,
            batch_size=256,
            pretrain_epoch=200,
            train_epoch=300,
            warmup_epoch=100,
            finetune_epoch=500,
            print_epoch=10,

            input_dims=[[240, 512, 128],
                        [216, 512, 128]],

            got=dict(
                lr=0.5,
                alpha=0.1,
                it=30,
                tau=2,
                num_iter=10,
                seed=10
            )
        )
    
    elif dataset == 'BDGP':
        return dict(
            lr=1e-4,
            aligned_ratio=0.5,
            beta=0.5,

            lambda_cons=1000,
            k=10,

            seed=9,
            print_epoch=10,
            pretrain_epoch=200,
            train_epoch=300,
            warmup_epoch=100,
            finetune_epoch=500,

            input_dims=[[1750, 256, 128],
                        [79, 256, 128]],

            got=dict(
                lr=0.5,
                alpha=0.1,
                it=30,
                tau=2,
                num_iter=5,
                seed=10
            )
        )

    elif dataset == 'Caltech101-7':
        return dict(
            lr=1e-4,
            aligned_ratio=0.5,
            beta=0.1,

            lambda_cons=10,
            k=30,

            seed=12 ,
            print_epoch=10,
            pretrain_epoch=200,
            train_epoch=400,
            warmup_epoch=200,
            finetune_epoch=300,
            input_dims=[[1984, 512, 256],
                        [512, 512, 256]],

            got=dict(
                lr=0.5,
                alpha=0.1,
                it=30,
                tau=2,
                num_iter=10,
                seed=10
            )
        )

    elif dataset == 'WebKB':
        return dict(
            lr=1e-4, 
            aligned_ratio=0.5,
            beta=0.5,

            lambda_cons=100,
            k=30,

            seed=1,

            pretrain_epoch=200,
            train_epoch=200,
            warmup_epoch=100,
            finetune_epoch=300,
            print_epoch=10,

            input_dims=[[2949, 512, 128],
                        [334, 512, 128]],

            got=dict(
                update=20,
                init_epoch=200,
                update_epoch=150,
                lr=0.5,
                alpha=0.5,
                it=30,
                tau=2,
                num_iter=10,
                seed=10
            )
        )

    elif dataset == 'Reuters':
        return dict(
            lr=1e-4,
            aligned_ratio=0.5,
            beta=0.1,

            lambda_cons=100,
            k=25,

            seed=5,

            pretrain_epoch=200,
            train_epoch=300,
            warmup_epoch=100,
            finetune_epoch=300,
            print_epoch=10,

            input_dims=[[10, 512, 256],
                        [10, 512, 256]],

            got=dict(
                update=20,
                init_epoch=200,
                update_epoch=150,
                lr=0.5,
                alpha=0.1,
                it=30,
                tau=2,
                num_iter=1,
                seed=10
            )
        )

    elif dataset == 'Fashion':
        return dict(
            lr=1e-4,
            aligned_ratio=0.5,
            beta=0.1,

            lambda_cons=0.1,
            k=30,

            seed=2,

            pretrain_epoch=200,
            train_epoch=400,
            warmup_epoch=200,
            finetune_epoch=300,
            print_epoch=10,

            input_dims=[[784, 512, 128],
                        [784, 512, 128]],

            got=dict(
                update=20,
                init_epoch=200,
                update_epoch=150,
                lr=0.5,
                alpha=0.5,
                it=30,
                tau=2,
                num_iter=1,
                seed=10
            )
        )

