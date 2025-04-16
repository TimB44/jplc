use crate::{
    asm_codegen::AsmEnv,
    ast::expr::{Expr, ExprKind},
};

impl AsmEnv<'_> {
    pub fn is_tensor(&self, expr: &Expr) -> bool {
        let (arr_loop_vars, arr_body, arr_scope) =
            if let ExprKind::ArrayComp(loop_vars, body, scope) = expr.kind() {
                (loop_vars, body, scope)
            } else {
                return false;
            };

        //TODO: check for all contants in loop_vars

        let (sum_loop_vars, sum_body, sum_scope) =
            if let ExprKind::Sum(loop_vars, body, scope) = arr_body.kind() {
                (loop_vars, body, scope)
            } else {
                return false;
            };
    }

    fn is_tensor_body(&self, expr: &Expr) {}
}

