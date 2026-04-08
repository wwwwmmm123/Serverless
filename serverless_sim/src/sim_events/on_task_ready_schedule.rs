use crate::{fn_dag::FnId, request::Request, sim_env::SimEnv};

impl SimEnv {
    pub fn on_task_ready_sche(&self, req: &mut Request, fnid: FnId) {
        let fnmetric = req.fn_metric.get_mut(&fnid).unwrap();
        
        // Check for duplicate ready_sche events
        if fnmetric.ready_sche_time.is_some() {
            log::warn!(
                "Task marked ready_sche multiple times: req={}, fn={}, prev_frame={}, current_frame={}",
                req.req_id,
                fnid,
                fnmetric.ready_sche_time.unwrap(),
                self.current_frame()
            );
            return; // Skip duplicate marking
        }
        
        fnmetric.ready_sche_time = Some(self.current_frame());
        assert!(fnmetric.data_recv_done_time.is_none());
        // Happend in this frame. So real ready is next frame
    }
}
