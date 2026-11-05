from trainkit.recipes import all_recipes, env_spec, get_recipe, recipe_for_task


def test_recipes_are_consistent():
    rs = all_recipes()
    assert [r.id for r in rs][0] == "yolo-detect"
    for r in rs:
        assert r.default_weights in r.weights["choices"], r.id
        assert r.smoke_dataset, r.id
        assert r.needs("train") and r.needs("preview")
        assert env_spec(r.env)["constraints_path"].exists()


def test_lookup():
    assert get_recipe("rtdetr").model_class == "RTDETR"
    assert recipe_for_task("segment").id == "yolo-seg"
